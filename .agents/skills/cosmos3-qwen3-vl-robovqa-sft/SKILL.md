---
name: cosmos3-qwen3-vl-robovqa-sft
description: >
  Fine-tune Qwen3-VL-8B-Instruct, a vision-language model from outside the
  Cosmos family, with the Cosmos Framework posttrain SFT path on robot-video
  question answering (RoboVQA), on one GPU, and score it before and after with
  the same evaluator. Use when the user asks to "fine-tune Qwen3-VL with Cosmos
  Framework", "train a non-Cosmos model or a VLM in the framework", "run SFT on
  robot videos / RoboVQA", "reproduce the Qwen3-VL RoboVQA result", or wants a
  worked post-training example with a measured before and after (environments,
  data preparation, preflight, training, Hugging Face export, evaluation). Not
  for Cosmos3 checkpoints (use cosmos3-post-training) and not for LoRA (the SFT
  path trains every parameter).
---

# Qwen3-VL-8B SFT on RoboVQA (one GPU)

One job: full-parameter SFT of `Qwen/Qwen3-VL-8B-Instruct` on the RoboVQA training clips through the framework's
posttrain SFT path, then the same scorer on 1,000 held-out clips before and after. Example from the held-out set:
"put the box on the table Q: possible right now?", reference `no`: the base model answers `yes`, the fine-tuned
model `no`.

Measured on one NVIDIA GB300 (aarch64, driver 595, CUDA 13.0), two runs at seed 0 (training is not
bit-reproducible): exact match 0.548 before, 0.632 and 0.620 after. Details:
[`references/measured-run.md`](references/measured-run.md).

## For an agent

The whole job is the eight steps below, run in order, each one command. A request that fits in one line:

> Using the `cosmos3-qwen3-vl-robovqa-sft` skill in this Cosmos Framework checkout: set up the environment,
> download the model and RoboVQA, prepare the training rows, run the preflight, score Qwen3-VL-8B before
> training, fine-tune for one epoch, export, and score after. Report both scores.

Rules: run the steps in the order listed; after each step compare the last lines with the step's "Expected";
stop at the first `FAILED` and report the step, the stage and the log path it printed; do not change the
commands or the recipe. With `--smoke` on every step the same request is a 25-minute check on six bundled clips
and 2 updates (the scores then mean nothing).

## Read this first

1. **One command per step.** Every step is
   `bash .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/run.sh <step>`, run from the repository root (the
   directory with `pyproject.toml`). The script sets its own paths and pins, so each step can run in a fresh shell;
   run the steps in order. [`scripts/run.sh`](scripts/run.sh) shows every command a step runs, with comments.
2. **Pick the variant from the GPU** (table below). The smoke variant adds `--smoke` to every step.
3. **Check each step's expected output** (under the step) before the next step. The screen shows one line per
   stage (`done` with its time) and a one-line result; each stage's full output goes to a file in the `logs` folder
   of the artifacts directory, which the script creates as artifacts/qwen3-vl-robovqa under the repository root
   (`-smoke` suffix for the smoke variant). A failing stage prints `FAILED`, the last 30 lines of its log and the
   log's path, and the step stops there.
4. **After a failure**, read "Things not obvious from the docs" and the troubleshooting section of
   [`references/measured-run.md`](references/measured-run.md) before changing a command.

## Your own data or another Qwen3-VL size

- **Another dataset.** The recipe reads `SFT_DATA_ROOT` (the `data` step writes it as `artifacts/qwen3-vl-robovqa/data`).
  Point it at your own folder and skip the `data` step. The folder needs `train.jsonl` in the framework's SFT
  conversation format (`docs/posttrain/guides/sft.md` section 3: one conversation per line, a `video` or `image`
  part plus the question in the user turn, the answer in the assistant turn, media paths relative to the file), and
  for the `before` and `after` steps `hf/test.jsonl` and `samples_test.jsonl` as `scripts/prepare_robovqa.py` writes
  them. The scorer's metrics (normalised exact match, token F1, yes/no accuracy) suit short-answer VQA; other tasks
  need their own scorer. Run `preflight` first: it refuses rows the recipe cannot train on.
- **Another Qwen3-VL size.** The model is one line of the recipe, `cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_robovqa.py`
  (`SFT_MODEL_PATH`, which the `download` step sets to the 8B snapshot). Copy the config, change the model, and
  pass the copy as `CONFIG` to `run.sh`; the framework's SFT path supports the Qwen3-VL and Qwen3-VL-MoE
  families (`docs/posttrain/guides/sft.md` section 1). Memory scales with size: 8B peaks at 143,566 MiB on one
  GPU. Only the 8B on RoboVQA is measured.
- **Another model family** (Llama, Gemma, InternVL, ...): not through this path; the SFT builder accepts the Qwen
  families only.

## When to use this skill

- Fine-tune Qwen3-VL-8B on RoboVQA with Cosmos Framework, or reproduce the before-and-after numbers above
- A worked, measured example of the posttrain SFT path for a Hugging Face VLM on video question answering
- Score a Hugging Face Qwen3-VL model on the RoboVQA held-out clips (steps `before` and `after`)
- Not for Cosmos3 models: `../cosmos3-post-training/SKILL.md`. Not for LoRA: the SFT path does not support it
  (`docs/posttrain/guides/sft.md` section 1). Environment errors: `../cosmos3-env-troubleshoot/SKILL.md`

## Variants and hardware

| Variant | Data | GPU | Time | Status |
| --- | --- | --- | --- | --- |
| Full run | RoboVQA, 7,998 training clips, 125 updates | one GPU with at least 150,000 MiB free for `train` (torch peaks at 143,566 MiB reserved); every other step fits a smaller GPU | about 3.5 h, training 2 h 05 to 2 h 14 | verified twice on one GB300, October 2026 |
| Smoke (`--smoke`) | the six bundled clips, 2 updates | `setup` to `before` on any GPU that holds the 8B model in bfloat16 (16.3 GiB); `train`, `export`, `after` need the same 150 GB GPU | environments plus about 15 minutes | measured by this skill's eval |

Disk: about 110 GB for the full run, about 75 GB for the smoke variant (environments, model, checkpoint, export).
x86_64 hosts take the `nvidia-npp` wheel from PyPI in `setup`; that route is unverified. On a host with several
GPUs the script uses the one with the most memory for every GPU step (it sets `CUDA_VISIBLE_DEVICES` unless the
caller already did). The bundled clips are Git LFS objects: `download --smoke` fetches them when the clone holds
pointer files, and asks for `git-lfs` to be installed when it is missing.

## Bundled assets

| Asset | Path | Used by |
| --- | --- | --- |
| Six RoboVQA clips with their source rows (Tianli/robovqa, Apache-2.0, revision `eb58194`) | `assets/robovqa/videos/`, `assets/robovqa/json/train/sample.json`, `assets/robovqa/LICENSE.txt` | every `--smoke` step |

The bundled directory has the layout of the full download. Six clips split 4 / 0 / 2 (train / val / test).

## Steps

### 1. setup — environments (about 12 minutes)

```bash
bash .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/run.sh setup
```

Four stages: the host check (CUDA 13 driver, `git`, `ffmpeg`, `g++`, uv; run as root it installs the missing tools
with apt-get itself, otherwise it prints the apt-get command and stops), the SFT environment from
`docs/posttrain/guides/sft.md` section 2, apex (the recipe's optimizer, compiled from source unless already
installed; the long stage, about 10 minutes), and the evaluation environment (plain Transformers). Expected last
lines: `SFT environment: 2.11.0+cu130 5.8.1 apex ok` and `evaluation environment: 2.10.0+cu130 4.57.6`. The
compiler's output is in the logs folder as apex-build.log (apex-build-smoke.log with `--smoke`).

Agents: this is one command that runs for about 12 minutes. Run it as one shell call with a long timeout, or start it
detached the way step 6 shows for `train` (`nohup bash .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/run.sh setup
> setup.out 2>&1 &`) and wait for its last lines before the next step. If your shell tool stops it early, run the same
command again: the finished environments and the apex build are kept, so the re-run continues from where it stopped.

### 2. download — model and data

```bash
bash .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/run.sh download
```

Qwen3-VL-8B-Instruct (17 GB) and RoboVQA (3.5 GB), both at fixed revisions; no token needed. With `--smoke` the
dataset stage instead makes sure the six bundled clips are real videos, not Git LFS pointers. Expected last line:
`model on disk: 17G   dataset: 9999 clips` (`6 clips` with `--smoke`).

### 3. data — training rows (about 25 minutes)

```bash
bash .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/run.sh data
```

[`scripts/prepare_robovqa.py`](scripts/prepare_robovqa.py) splits the clips 80/10/10 with one question per clip and
byte-identical clips kept in one split, writes each training clip's evaluation frames losslessly so training sees
exactly what the evaluator scores, and checks every file in the training stack. Expected last lines: the counts
`"train": 7998, "val": 1001, "test": 1000` with every `leakage_check` value 0, and `frame check: "all_ok": true,
"bad": 0`. With `--smoke`: `"train": 4, "val": 0, "test": 2`. The split log also prints the md5 of the samples
files next to their seed-0 references; they match on the full download only.

### 4. preflight — the recipe's CPU checks (about 13 minutes)

```bash
bash .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/run.sh preflight
```

`--validate-manifest` and `--validate-rows` of the SFT entrypoint with the recipe
`cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_robovqa.py`. Expected last line:
`train: rows 7998 | loss_tokens == assistant_tokens on 7998/7998 | longest 4113 <= 10240 | failed 0` (`rows 4` with
`--smoke`). The log's `train: dp_world not checked` line is expected outside torchrun.

### 5. before — score the untouched model (about 14 minutes)

```bash
bash .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/run.sh before
```

[`scripts/eval_robovqa.py`](scripts/eval_robovqa.py) answers every held-out clip greedily and scores normalised exact
match, token F1 and yes/no accuracy. Expected last line: `exact 0.548  token-F1 0.676  yes/no 0.918 (n=1000)`
(`n=2` with `--smoke`).

### 6. train — fine-tune (about 2 hours)

```bash
bash .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/run.sh train
```

One GPU through torchrun, 125 updates of 64 clips (2 with `--smoke`). It stops before starting when the GPU has less
than 150,000 MiB free. The screen shows one line per update, `step N/125  train/loss_avg …  … s`, the first after
about three minutes and then about one a minute; the loss starts near 1.0 and ends near 0.5. Expected end:
`SFT DONE {'start': 0, 'final_step': 125, ...}`. The framework's full log is train.log in the logs folder. Agents: start it
detached, for example `nohup bash .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/run.sh train > train.out 2>&1 &`.

### 7. export — Hugging Face format (about 2 minutes)

```bash
bash .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/run.sh export
```

The framework's HF materializer on the CPU, in 5 GB shards. Expected last line:
`[materialize] OK -> .../export (16.3 GiB, 750 tensors in 4 shards, 8 sidecars identical to base, ...)`.

### 8. after — score the fine-tuned model (about 11 minutes)

```bash
bash .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/run.sh after
```

Expected: exact match near 0.62 to 0.63 (0.632 and 0.620 in the two measured runs). Near 0.55 means the fine-tuned
weights were not loaded. With `--smoke` the score is only a check that the export loads and answers.

## Verify the output

Every stage's full output is in the artifacts directory's `logs` folder, named after the stage (`setup-sft-env`,
`apex-build`, `data-split`, `data-frames`, `data-verify`, `preflight-rows`, `before`, `train`, `export`, `after`, …;
`-smoke` suffix for the smoke variant; a re-run of a stage keeps the previous attempt's log as `<name>.log.prev`). A run
is correct when:

| Step | Correct when |
| --- | --- |
| data | the split manifest's `leakage_check` values are all 0 and verify prints `"all_ok": true, "bad": 0` |
| preflight | the train gate line ends in `failed 0` |
| before, after | the overall line reports `n=1000` (`n=2` with `--smoke`) and the scores file shows no missing predictions |
| train | the log has a `step N/125` line per update and `SFT DONE` |
| export | `[materialize] OK` with 750 tensors |

## Reference files

| File | What it holds |
| --- | --- |
| [`scripts/run.sh`](scripts/run.sh) | every step's commands, with comments |
| [`references/measured-run.md`](references/measured-run.md) | the two measured runs: versions, timings, memory, loss, example answers, troubleshooting |
| `cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_robovqa.py` | the recipe, each field with its reason |
| [`scripts/prepare_robovqa.py`](scripts/prepare_robovqa.py) | split, frames, verify |
| [`scripts/eval_robovqa.py`](scripts/eval_robovqa.py) | evaluator, scorer and the shared frame decoder |

## Things not obvious from the docs

- **`could not read the CUDA version from the nvidia-smi banner`**: the host check reads the banner's `CUDA Version:`
  label (drivers up to R595) or `CUDA UMD Version:` (R615 and later, where the old label is deprecated). Another label
  is a format this skill has not seen: check that `nvidia-smi -q | grep -i 'cuda.*version'` reports 13 or newer, then
  open an issue with the banner line.
- **`libnppicc.so.13: cannot open shared object file`**: TorchCodec needs NVIDIA NPP; `setup` installs it and every
  step that decodes video points `LD_LIBRARY_PATH` at it. Do not clear `LD_LIBRARY_PATH`.
- **`ImportError: libc10.so` importing `amp_C`**: import `torch` first; apex's extensions link against torch.
- **apex gone after re-running `uv sync` by hand**: an exact sync uninstalls packages the lock does not list;
  `setup` syncs with `--inexact`.
- **SFT environment on Python 3.12**: TorchCodec then cannot find NPP; `setup` pins `--python 3.13`.
- **`[materialize] FATAL ... model.safetensors.index.json missing`**: a framework checkout older than the 5 GB shard
  change; Transformers 5 writes a model under 50 GB as one file without the index the verifier requires.
- **`hf download` fetched no videos**: a newer `hf`; `download` uses the evaluation environment's (huggingface_hub
  0.36.2).
- **The first training update takes about two minutes and nothing prints before it**: model load and warm-up.
- **Running `train` or `export` a second time in the same checkout**: the framework refuses to start at step 0 in a
  run directory that already holds a checkpoint, and the materializer refuses an existing output. Both steps move
  the earlier result aside under a timestamped name and say so; nothing is deleted.

## Related skills

| Skill | When to use |
| --- | --- |
| `../cosmos3-setup/SKILL.md` | framework installation and checkpoint downloads in general |
| `../cosmos3-post-training/SKILL.md` | post-training Cosmos3 models |
| `../cosmos3-env-troubleshoot/SKILL.md` | import, CUDA, shared-library and download errors |
