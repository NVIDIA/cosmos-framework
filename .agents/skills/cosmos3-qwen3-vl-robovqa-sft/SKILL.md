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

One job: full-parameter SFT of `Qwen/Qwen3-VL-8B-Instruct` on the RoboVQA training clips through
`cosmos_framework.posttrain.entrypoints.sft`, then the same scorer on 1,000 held-out clips before and after.
Example from the held-out set: the clip of a robot arm by a table, "put the box on the table Q: possible right
now?", reference `no`: the base model answers `yes`, the fine-tuned model `no`.

Measured on one NVIDIA GB300 (aarch64, driver 595, CUDA 13.0), two runs at seed 0 (training is not
bit-reproducible): exact match 0.548 before, 0.632 and 0.620 after. Details:
[`references/measured-run.md`](references/measured-run.md).

## Read this first

1. **Path convention.** Run every block from the repository root (the directory with `pyproject.toml`). Every
   block starts with `source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh`, which sets the paths,
   the uv pin and the artifact directory `$ART`; no block relies on shell state from an earlier block.
2. **Pick the variant from the GPU** (table below): the full run needs one GPU with about 150 GB free; the smoke
   variant (steps 0 to 5 on the bundled clips) runs on any GPU that holds the 8B model in bfloat16.
3. **Two Python environments, never mixed**: the SFT environment (framework, Transformers 5.8.1, TorchCodec 0.11)
   prepares the training files, checks and trains; the evaluation environment (plain Transformers 4.57.6,
   TorchCodec 0.10) decodes the evaluation frames and scores. Each command names its interpreter.
4. **Check each step's expected output** (given under the step and in "Verify the output") before the next step.
5. **After a failure**, read "Things not obvious from the docs" and
   [`references/measured-run.md`](references/measured-run.md) § Troubleshooting before changing a command.

## When to use this skill

- Fine-tune Qwen3-VL-8B on RoboVQA with Cosmos Framework, or reproduce the before-and-after numbers above
- A worked, measured example of the posttrain SFT path for a Hugging Face VLM on video question answering
- Score any Hugging Face Qwen3-VL directory on the RoboVQA held-out clips (step 5 / step 8)
- Not for Cosmos3 models: `../cosmos3-post-training/SKILL.md`. Not for LoRA: the SFT path does not support it
  (`docs/posttrain/guides/sft.md` section 1). Environment errors: `../cosmos3-env-troubleshoot/SKILL.md`

## Variants and hardware

| Variant | GPU | Steps | Time | Status |
| --- | --- | --- | --- | --- |
| Full run | one GPU with at least 150,000 MiB free; torch peaks at 143,566 MiB reserved, so 80 to 141 GB cards do not fit | 0 to 8 | about 3.5 h (training 2 h 05 to 2 h 14) | verified twice on one GB300, October 2026 |
| Smoke | any GPU for step 5 (the 8B model in bfloat16, 16.3 GiB of weights); steps 3 and 4 run on the CPU | 0 to 5 on `assets/robovqa` | environments plus a few minutes | the same commands on 6 bundled clips; measured by this skill's eval only |

Disk: about 110 GB for the full run (frames 33 GB, model, checkpoint and export 17 GB each), about 40 GB for the
smoke variant (environments and the model). The x86_64 branches below (`nvidia-npp` from PyPI) are unverified.
To score a model without training: steps 0 and 1c, the model download of step 2, the `split` line of step 3 and
step 5.

## Bundled assets

| Asset | Path | Used by |
| --- | --- | --- |
| Six RoboVQA clips with their source rows (Tianli/robovqa, Apache-2.0, revision `eb58194`) | `assets/robovqa/videos/`, `assets/robovqa/json/train/sample.json`, `assets/robovqa/LICENSE.txt` | smoke variant, steps 3 to 5 |

The bundled directory has the layout of the full download, so the smoke commands differ from the full ones in
their `--src` / `--out` paths only. Six clips split 4 / 0 / 2 (train / val / test).

## Step 0 — prerequisites

Check the GPU, the driver's CUDA version (the wheels are CUDA 13.0), the system tools and uv:

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader
CUDA_MAJOR=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p' | head -1)
[ "${CUDA_MAJOR:-0}" -ge 13 ] || { echo "driver supports CUDA ${CUDA_MAJOR}; this recipe uses CUDA 13.0 wheels (a CUDA 12 route is unverified)"; exit 1; }
MISSING=""
for tool in git ffmpeg g++ curl; do command -v "$tool" >/dev/null || MISSING="$MISSING $tool"; done
if [ -n "$MISSING" ]; then
  if [ "$(id -u)" = 0 ]; then apt-get update -q && apt-get install -y -q git ffmpeg g++ curl
  else echo "missing:$MISSING (sudo apt-get install -y git ffmpeg g++ curl)"; exit 1; fi
fi
command -v uvx >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
$UV --version
ffmpeg -version | head -1
df -h "$ART" | tail -1
```

Continue when the CUDA check passes, `uv 0.11.29` prints, `ffmpeg` is 4 to 8, and the free disk covers the
variant (above).

## Step 1 — environments

**1a. SFT environment** — `docs/posttrain/guides/sft.md` section 2 plus `--python 3.13` (with `--project`, uv does
not read the checkout's `.python-version` and can pick a system 3.12), the aarch64 group that adds `nvidia-npp`
(TorchCodec needs `libnppicc.so.13`) and `--inexact` (an exact sync uninstalls apex, built in 1b):

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
ARCH_GROUP=""; [ "$(uname -m)" = aarch64 ] && ARCH_GROUP="--group cu130-aarch64"
UV_PROJECT_ENVIRONMENT="$SFT_VENV" $UV sync \
  --project docker/posttrain/environment \
  --frozen \
  --python 3.13 \
  --no-group transformers4 \
  --group sft \
  --group cu130-runtime \
  $ARCH_GROUP \
  --inexact
[ "$(uname -m)" = aarch64 ] || $UV pip install --python "$SFT_PY" "nvidia-npp==13.0.0.50"
LD_LIBRARY_PATH="$SFT_LIBS" "$SFT_PY" -c "import torch, transformers, torchcodec.decoders; print(torch.__version__, transformers.__version__)"
```

Expected: `2.11.0+cu130 5.8.1`.

**1b. apex** — the native Qwen3-VL seat trains with apex `FusedAdam`, and the SFT preflight refuses `fusedadam`
without apex. NVIDIA training containers ship it; otherwise this builds it from source with the CUDA 13.0 compiler
and headers from PyPI (about 10 minutes on a GB300), for the GPU's own architecture, with one compile job per
4 GiB of free host memory:

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
if "$SFT_PY" -c "import torch, amp_C" 2>/dev/null; then
  echo "apex already importable"
else
  TC="$ART/venvs/cuda13-toolchain"
  $UV venv --allow-existing --python 3.13 "$TC"
  $UV pip install --python "$TC/bin/python" ninja \
    "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-cccl==13.0.85" "nvidia-cuda-crt==13.0.88" "nvidia-nvvm==13.0.88" \
    "nvidia-cuda-runtime==13.0.96" "nvidia-cuda-profiler-api==13.0.85" "nvidia-cublas==13.1.0.3" \
    "nvidia-cusparse==12.6.3.3" "nvidia-cusolver==12.0.4.66" "nvidia-curand==10.4.0.35" \
    "nvidia-cufft==12.0.0.61" "nvidia-nvjitlink==13.0.88"
  export CUDA_HOME="$TC/lib/python3.13/site-packages/nvidia/cu13"
  (cd "$CUDA_HOME/lib" && for f in libcudart libcublas libcublasLt libcusparse libcusolver libcurand libcufft libnvJitLink; do
    ln -sfn "$(ls $f.so.* | head -1)" "$f.so"; done)
  ln -sfn lib "$CUDA_HOME/lib64"
  export PATH="$CUDA_HOME/bin:$TC/bin:$PATH"
  [ -d "$ART/apex/.git" ] || git clone https://github.com/NVIDIA/apex.git "$ART/apex"
  git -C "$ART/apex" checkout --detach 575968bc1f9127ccd61003a681472a83af4ff1a1
  JOBS=$(awk -v n="$(nproc)" '/MemAvailable/ {m = int($2 / 4194304); print (m < n ? m : n)}' /proc/meminfo)
  TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)" \
  APEX_CPP_EXT=1 APEX_CUDA_EXT=1 MAX_JOBS="$JOBS" \
    $UV pip install --python "$SFT_PY" --no-build-isolation "$ART/apex"
fi
LD_LIBRARY_PATH="$SFT_LIBS" "$SFT_PY" -c "import torch, transformers, torchcodec.decoders, amp_C; print('apex ok')"
```

Expected: `apex ok`. (The GB300 runs built with 64 jobs; the memory-based cap is a recommendation for smaller hosts.)

**1c. Evaluation environment** — plain Transformers, the stack the scores were measured with:

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
$UV venv --allow-existing --python 3.12 "$EVAL_VENV"
$UV pip install --python "$EVAL_PY" "torch==2.10.0" "torchvision==0.25.0" "torchcodec==0.10.0" \
  --index-url https://download.pytorch.org/whl/cu130
$UV pip install --python "$EVAL_PY" "transformers==4.57.6" "accelerate==1.15.0" "av==19.0.1" "numpy==2.2.6" \
  pillow safetensors "huggingface_hub==0.36.2" "nvidia-npp==13.0.0.50"
LD_LIBRARY_PATH="$EVAL_LIBS" "$EVAL_PY" -c "import torch, transformers, torchcodec.decoders; print(torch.__version__, transformers.__version__)"
```

Expected: `2.10.0+cu130 4.57.6`.

## Step 2 — model and data

Both pinned, downloaded with the evaluation environment's `hf` (huggingface_hub 0.36.2; newer `hf` releases read
the patterns after the first `--include` as file names and skip the videos folder). No token is needed.

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
"$EVAL_VENV/bin/hf" download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir "$SFT_MODEL_PATH"
```

**Full run only** (3.5 GB; the smoke variant uses the bundled clips instead):

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
"$EVAL_VENV/bin/hf" download Tianli/robovqa --repo-type dataset \
  --revision eb5819490f0d9ec18a79774233d4b1c9ebd52eea \
  --include "videos/*" "json/*" "LICENSE.txt" \
  --local-dir "$ART/robovqa"
```

## Step 3 — training rows

[`scripts/prepare_robovqa.py`](scripts/prepare_robovqa.py): `split` (80/10/10 by clip, one question per clip,
byte-identical clips kept in one split), `frames` (each training clip's evaluation frames, every round(fps/2)-th
frame up to 32, written losslessly so training sees exactly what the evaluator decodes; evaluation environment),
`verify` (the training stack decodes the same frames and the processor gives the same tokens and pixels; SFT
environment). Full run, about 25 minutes with 48 CPU processes:

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
"$EVAL_PY" "$SKILL_DIR/scripts/prepare_robovqa.py" split --src "$ART/robovqa" --out "$ART/data"
CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH="$EVAL_LIBS" \
  "$EVAL_PY" "$SKILL_DIR/scripts/prepare_robovqa.py" frames --out "$ART/data"
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 LD_LIBRARY_PATH="$SFT_LIBS" \
  "$SFT_PY" "$SKILL_DIR/scripts/prepare_robovqa.py" verify --out "$ART/data" --model "$SFT_MODEL_PATH"
```

Expected: counts `train 7998, val 1001, test 1000`; the md5 lines equal their seed-0 references
(`0b3739a5…` train, `1a9b1924…` test); `frames` ends with `"rows": 7998` and `"T_total": 119827`; `verify` ends with
`"all_ok": true, "bad": 0` and `"len_max": 4113`.

**Smoke variant** (the bundled clips; the md5 references apply to the full download only):

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
"$EVAL_PY" "$SKILL_DIR/scripts/prepare_robovqa.py" split --src "$SKILL_DIR/assets/robovqa" --out "$ART/smoke"
CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH="$EVAL_LIBS" \
  "$EVAL_PY" "$SKILL_DIR/scripts/prepare_robovqa.py" frames --out "$ART/smoke"
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 LD_LIBRARY_PATH="$SFT_LIBS" \
  "$SFT_PY" "$SKILL_DIR/scripts/prepare_robovqa.py" verify --out "$ART/smoke" --model "$SFT_MODEL_PATH"
```

Expected: counts `train 4, val 0, test 2`; `frames` `"rows": 4`; `verify` `"all_ok": true, "bad": 0`.

## Step 4 — preflight

The two CPU checks of `docs/posttrain/guides/sft.md` section 2 with the recipe that trains,
`cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_robovqa.py` (it reads `SFT_MODEL_PATH` and `SFT_DATA_ROOT`).
Full run, about 13 minutes:

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
for check in --validate-manifest --validate-rows; do
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. SFT_DATA_ROOT="$ART/data" IMAGINAIRE_OUTPUT_ROOT="$ART/runs/preflight" \
  LD_LIBRARY_PATH="$SFT_LIBS" \
    "$SFT_PY" -m cosmos_framework.posttrain.entrypoints.sft --config "$CONFIG" "$check" || exit 1
done 2>&1 | tee "$ART/logs/preflight.log"
```

Expected gate line:
`train: rows 7998 | loss_tokens == assistant_tokens on 7998/7998 | longest 4113 <= 10240 | failed 0`; the next
line, `train: dp_world not checked`, is expected outside torchrun.

**Smoke variant:**

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
for check in --validate-manifest --validate-rows; do
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. SFT_DATA_ROOT="$ART/smoke" IMAGINAIRE_OUTPUT_ROOT="$ART/runs/preflight-smoke" \
  LD_LIBRARY_PATH="$SFT_LIBS" \
    "$SFT_PY" -m cosmos_framework.posttrain.entrypoints.sft --config "$CONFIG" "$check" || exit 1
done 2>&1 | tee "$ART/logs/preflight-smoke.log"
```

Expected gate line: `train: rows 4 | loss_tokens == assistant_tokens on 4/4 | ... | failed 0`.

## Step 5 — before: score the untouched model

[`scripts/eval_robovqa.py`](scripts/eval_robovqa.py) answers greedily (at most 64 new tokens, batches of 4,
bfloat16) and scores normalised exact match, token F1 and yes/no accuracy. Full run, about 14 minutes:

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
LD_LIBRARY_PATH="$EVAL_LIBS" "$EVAL_PY" "$SKILL_DIR/scripts/eval_robovqa.py" \
  --model "$SFT_MODEL_PATH" \
  --data "$ART/data/hf/test.jsonl" \
  --ref "$ART/data/samples_test.jsonl" \
  --out "$ART/eval/zeroshot"
```

Expected last line: `exact 0.548  token-F1 0.676  yes/no 0.918 (n=1000)` (deterministic).

**Smoke variant** (2 held-out clips; a pipeline check, not a quality measure):

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
LD_LIBRARY_PATH="$EVAL_LIBS" "$EVAL_PY" "$SKILL_DIR/scripts/eval_robovqa.py" \
  --model "$SFT_MODEL_PATH" \
  --data "$ART/smoke/hf/test.jsonl" \
  --ref "$ART/smoke/samples_test.jsonl" \
  --out "$ART/eval/zeroshot-smoke"
```

Expected: `$ART/eval/zeroshot-smoke/preds.jsonl` with 2 rows, `$ART/eval/zeroshot-smoke/scores.json`, and a last line
`exact … token-F1 … yes/no … (n=2)`.

## Step 6 — train (full run only)

One GPU through torchrun, as `docs/posttrain/guides/sft.md` requires; started detached with its log kept. The block
stops before launching when the GPU has less than 150,000 MiB free:

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$FREE" -ge 150000 ] || { echo "${FREE} MiB free; full SFT of the 8B model needs about 150 GB (measured peak 143,566 MiB reserved)"; exit 1; }
PYTHONPATH=. SFT_DATA_ROOT="$ART/data" IMAGINAIRE_OUTPUT_ROOT="$ART/runs" LD_LIBRARY_PATH="$SFT_LIBS" \
  setsid nohup "$SFT_PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  -m cosmos_framework.posttrain.entrypoints.sft \
  --config "$CONFIG" \
  > "$ART/logs/train.log" 2>&1 < /dev/null &
echo "training started; log: $ART/logs/train.log"
```

Wait for it (about three minutes to update 1, then one `step N/125` line about every minute; deadline 3 h):

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
for minute in $(seq 1 180); do
  grep -q "SFT DONE" "$ART/logs/train.log" && break
  pgrep -f "entrypoints.sft --config $CONFIG" >/dev/null || { echo "training exited"; tail -n 40 "$ART/logs/train.log"; exit 1; }
  sleep 60
done
grep -E "step +(1|125)/125 |SFT DONE" "$ART/logs/train.log" | cut -c1-120
```

Expected: the logged training loss (`loss_avg`) near 1.0 at update 1 and near 0.5 at update 125 (single updates
move with each batch of 64 clips), then `SFT DONE {'start': 0, 'final_step': 125, ...}`. A second seed: append `-- rl.seed=1 job.name=seed1`
to the launch command.

## Step 7 — export to Hugging Face format

The framework's HF materializer, on the CPU (about 2 minutes); it writes 5 GB shards and verifies all 750 tensors:

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
CKPT=$(ls -d "$ART"/runs/posttrain_sft/qwen3_vl_8b_robovqa/seed0/*/checkpoint_artifacts/iter_* | sort | tail -1)
CUDA_VISIBLE_DEVICES= PYTHONPATH=. LD_LIBRARY_PATH="$SFT_LIBS" \
  "$SFT_PY" -m cosmos_framework.posttrain.checkpoint.tools.materialize_hf \
  --ckpt-dir "$CKPT" \
  --base "$SFT_MODEL_PATH" \
  --out "$ART/export" \
  --expect-tensors 750 \
  --min-tensors 750
```

Expected: `[materialize] OK -> .../export (16.3 GiB, 750 tensors in 4 shards, 8 sidecars identical to base, ...)`.

## Step 8 — after: score the fine-tuned model

```bash
source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
LD_LIBRARY_PATH="$EVAL_LIBS" "$EVAL_PY" "$SKILL_DIR/scripts/eval_robovqa.py" \
  --model "$ART/export" \
  --data "$ART/data/hf/test.jsonl" \
  --ref "$ART/data/samples_test.jsonl" \
  --out "$ART/eval/sft"
```

Expected: exact match near 0.62 to 0.63 (0.632 and 0.620 in the two measured runs). A score near 0.55 means the
fine-tuned weights were not loaded.

## Verify the output

| Step | File | Correct when |
| --- | --- | --- |
| 3 | `$ART/data/split_manifest.json` | counts 7998 / 1001 / 1000; every `leakage_check` value is 0 |
| 3 | `$ART/data/train.jsonl.verify.jsonl` | 7,998 lines, all `sha_ok`, `ids_equal`, `grid_equal`, `pix_equal` true |
| 4 | `$ART/logs/preflight.log` | the gate line ends in `failed 0` |
| 5, 8 | `$ART/eval/<name>/scores.json` | `__all__.n` is 1000 and `__meta__.missing_predictions` is 0 |
| 6 | `$ART/logs/train.log` | 125 `step N/125` lines and `SFT DONE` |
| 7 | `$ART/export/model.safetensors.index.json` | 750 entries in `weight_map` |

Smoke variant: the same files under `$ART/smoke`, `$ART/logs/preflight-smoke.log` and `$ART/eval/zeroshot-smoke`,
with 4 training rows and `n` = 2.

## Reference files

| File | What it holds |
| --- | --- |
| [`references/measured-run.md`](references/measured-run.md) | the two measured runs: versions, timings, memory, loss, example answers, troubleshooting |
| `cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_robovqa.py` | the recipe, each field with its reason |
| [`scripts/prepare_robovqa.py`](scripts/prepare_robovqa.py) | split, frames, verify |
| [`scripts/eval_robovqa.py`](scripts/eval_robovqa.py) | evaluator, scorer and the shared frame decoder |
| [`scripts/env.sh`](scripts/env.sh) | paths and pins every block sources |

## Things not obvious from the docs

- **`libnppicc.so.13: cannot open shared object file`**: TorchCodec needs NVIDIA NPP. Every command that decodes
  video sets `LD_LIBRARY_PATH` to its environment's NPP library directory (`$SFT_LIBS` or `$EVAL_LIBS`). Do not clear
  `LD_LIBRARY_PATH`.
- **`ImportError: libc10.so` importing `amp_C`**: import `torch` first; apex's extensions link against torch.
- **apex gone after a second `uv sync`**: an exact sync uninstalls packages the lock does not list; keep `--inexact`.
- **SFT environment on Python 3.12**: `$SFT_LIBS` then points at a missing `python3.13` path; keep `--python 3.13`.
- **`[materialize] FATAL ... model.safetensors.index.json missing`**: a checkout whose materializer still writes one
  50 GB shard; Transformers 5 writes a model under 50 GB as one file without the index the verifier requires. The
  materializer now writes 5 GB shards.
- **`hf download` fetched no videos**: a newer `hf`; use `$EVAL_VENV/bin/hf` (huggingface_hub 0.36.2).
- **Training needs one GPU with about 150 GB**: 80 GB and 141 GB cards run every step except step 6.
- **The first update takes about two minutes and prints nothing before it**: model load and warm-up, not a hang.

## Related skills

| Skill | When to use |
| --- | --- |
| `../cosmos3-setup/SKILL.md` | framework installation and checkpoint downloads in general |
| `../cosmos3-post-training/SKILL.md` | post-training Cosmos3 models |
| `../cosmos3-env-troubleshoot/SKILL.md` | import, CUDA, shared-library and download errors |
