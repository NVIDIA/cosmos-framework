#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Fine-tune Qwen3-VL-8B-Instruct on RoboVQA with Cosmos Framework, one step per command.
# Run from the cosmos-framework repository root:
#
#   bash .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/run.sh <step> [--smoke]
#
# Steps, in order:
#   setup      the SFT environment, apex and the evaluation environment
#   download   Qwen3-VL-8B-Instruct and RoboVQA, at fixed revisions
#   data       split RoboVQA, write the training frames, check them
#   preflight  the SFT path's CPU checks of the recipe
#   before     score the untouched model on the held-out clips
#   train      fine-tune: one epoch, 125 updates of 64 clips, one GPU with about 150 GB
#   export     write the trained weights as a Hugging Face model
#   after      score the fine-tuned model the same way
#
# --smoke uses the skill's six bundled clips and 2 training updates: a quick check of every step.
# Every artifact goes under artifacts/qwen3-vl-robovqa/; each step's output is also saved under its logs/.
set -euo pipefail

STEP=${1:-}
SMOKE=0
[ "${2:-}" = "--smoke" ] && SMOKE=1
if [ ! -f pyproject.toml ] || [ ! -d cosmos_framework ]; then
  echo "run.sh: run it from the cosmos-framework repository root" >&2
  exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
SKILL_DIR="$PWD/.agents/skills/cosmos3-qwen3-vl-robovqa-sft"
UV="uvx --from uv==0.11.29 uv"  # the checkout pins uv 0.11.29
ART="$PWD/artifacts/qwen3-vl-robovqa"
SFT_VENV="$PWD/docker/posttrain/environment/.venv-sft"
SFT_PY="$SFT_VENV/bin/python"
SFT_LIBS="$SFT_VENV/lib/python3.13/site-packages/nvidia/cu13/lib"    # NVIDIA NPP, needed by TorchCodec
EVAL_VENV="$ART/venvs/eval"
EVAL_PY="$EVAL_VENV/bin/python"
EVAL_LIBS="$EVAL_VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
CONFIG="cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_robovqa.py"
export SFT_MODEL_PATH="$ART/models/Qwen3-VL-8B-Instruct"  # read by the recipe
if [ "$SMOKE" = 1 ]; then
  SRC="$SKILL_DIR/assets/robovqa" DATA="$ART/smoke" RUN="$ART/runs-smoke" EVAL="$ART/eval-smoke"
  EXPORT="$ART/export-smoke" STEPS=2 TAG="-smoke"
else
  SRC="$ART/robovqa" DATA="$ART/data" RUN="$ART/runs" EVAL="$ART/eval" EXPORT="$ART/export" STEPS=125 TAG=""
fi
export SFT_DATA_ROOT="$DATA"  # read by the recipe
mkdir -p "$ART/logs"

say() { printf '\n==> %s\n' "$*"; }

# stage <label> <log name> <command...>: one line on screen (the label, then "done" with the elapsed time), the
# command's whole output in $ART/logs/<log name>.log. On failure the last 30 lines of the log are shown.
stage() {
  local label=$1 log="$ART/logs/$2$TAG.log" start
  shift 2
  start=$(date +%s)
  printf '  %-62s' "$label"
  if "$@" >"$log" 2>&1; then
    printf 'done  %3d s\n' $(( $(date +%s) - start ))
  else
    printf 'FAILED after %d s\n\n' $(( $(date +%s) - start ))
    tail -n 30 "$log"
    echo
    echo "full output: $log" >&2
    exit 1
  fi
}

check_host() {
  local cuda missing=""
  cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p' | head -1)
  if [ "${cuda:-0}" -lt 13 ]; then
    echo "the driver supports CUDA ${cuda}; this example uses CUDA 13.0 wheels" >&2
    return 1
  fi
  for tool in git ffmpeg g++ curl; do command -v "$tool" >/dev/null || missing="$missing $tool"; done
  if [ -n "$missing" ]; then
    if [ "$(id -u)" = 0 ]; then apt-get update -q && apt-get install -y -q git ffmpeg g++ curl; else
      echo "install$missing first: sudo apt-get install -y git ffmpeg g++ curl" >&2
      return 1
    fi
  fi
  command -v uvx >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
}

sft_env() {
  # --python 3.13: with --project, uv does not read the checkout's .python-version.
  # --group cu130-aarch64 adds NVIDIA NPP on aarch64. --inexact keeps apex when this is re-run.
  local arch=()
  [ "$(uname -m)" = aarch64 ] && arch=(--group cu130-aarch64)
  UV_PROJECT_ENVIRONMENT="$SFT_VENV" $UV sync --project docker/posttrain/environment --frozen --python 3.13 \
    --no-group transformers4 --group sft --group cu130-runtime "${arch[@]}" --inexact
  [ "$(uname -m)" = aarch64 ] || $UV pip install --python "$SFT_PY" "nvidia-npp==13.0.0.50"  # x86_64: unverified
}

eval_env() {
  $UV venv --allow-existing --python 3.12 "$EVAL_VENV"
  $UV pip install --python "$EVAL_PY" torch==2.10.0 torchvision==0.25.0 torchcodec==0.10.0 \
    --index-url https://download.pytorch.org/whl/cu130
  $UV pip install --python "$EVAL_PY" transformers==4.57.6 accelerate==1.15.0 av==19.0.1 numpy==2.2.6 \
    pillow safetensors huggingface_hub==0.36.2 nvidia-npp==13.0.0.50
}

setup() {
  say "setup: two Python environments and the apex optimizer (about 12 minutes on a fresh machine)"
  echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
  stage "host: CUDA 13 driver, git, ffmpeg, g++, uv" setup-host check_host
  stage "SFT environment: torch 2.11, transformers 5.8.1, torchcodec 0.11 (about 270 packages)" setup-sft-env sft_env
  if "$SFT_PY" -c "import torch, amp_C" 2>/dev/null; then
    printf '  %-62s%s\n' "apex" "already installed"
  else
    stage "apex: compiled from source for this GPU (about 10 minutes)" apex-build build_apex
  fi
  stage "evaluation environment: torch 2.10, transformers 4.57.6, torchcodec 0.10" setup-eval-env eval_env
  echo
  LD_LIBRARY_PATH="$SFT_LIBS" "$SFT_PY" -c \
    "import torch, transformers, torchcodec.decoders, amp_C; print('  SFT environment:       ', torch.__version__, transformers.__version__, 'apex ok')"
  LD_LIBRARY_PATH="$EVAL_LIBS" "$EVAL_PY" -c \
    "import torch, transformers, torchcodec.decoders; print('  evaluation environment:', torch.__version__, transformers.__version__)"
  echo "  logs: $ART/logs/setup-*.log, $ART/logs/apex-build.log"
}

build_apex() {
  # The CUDA 13.0 compiler and headers come from PyPI, so the host needs no CUDA toolkit; about 10 minutes.
  local tc="$ART/venvs/cuda13-toolchain"
  $UV venv --allow-existing --python 3.13 "$tc"
  $UV pip install --python "$tc/bin/python" ninja \
    nvidia-cuda-nvcc==13.0.88 nvidia-cuda-cccl==13.0.85 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88 \
    nvidia-cuda-runtime==13.0.96 nvidia-cuda-profiler-api==13.0.85 nvidia-cublas==13.1.0.3 \
    nvidia-cusparse==12.6.3.3 nvidia-cusolver==12.0.4.66 nvidia-curand==10.4.0.35 nvidia-cufft==12.0.0.61 \
    nvidia-nvjitlink==13.0.88
  export CUDA_HOME="$tc/lib/python3.13/site-packages/nvidia/cu13"
  (cd "$CUDA_HOME/lib" && for f in libcudart libcublas libcublasLt libcusparse libcusolver libcurand libcufft libnvJitLink; do
    ln -sfn "$(ls $f.so.* | head -1)" "$f.so"
  done)
  ln -sfn lib "$CUDA_HOME/lib64"
  export PATH="$CUDA_HOME/bin:$tc/bin:$PATH"
  [ -d "$ART/apex/.git" ] || git clone https://github.com/NVIDIA/apex.git "$ART/apex"
  git -C "$ART/apex" checkout --detach 575968bc1f9127ccd61003a681472a83af4ff1a1
  # One compile job per 4 GiB of free host memory, at most 32 and at most one per CPU; built for this GPU.
  local jobs arch
  jobs=$(awk -v n="$(nproc)" '/MemAvailable/ {m = int($2 / 4194304); if (m > n) m = n; if (m > 32) m = 32; print m}' /proc/meminfo)
  arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  echo "compiling apex for compute capability $arch with $jobs parallel jobs"
  # Running setup.py directly (not through pip) keeps the compiler's output, which the stage writes to the log.
  $UV pip install --python "$SFT_PY" wheel  # bdist_wheel needs it; not in the SFT lock
  (cd "$ART/apex" && TORCH_CUDA_ARCH_LIST="$arch" APEX_CPP_EXT=1 APEX_CUDA_EXT=1 MAX_JOBS="$jobs" \
    "$SFT_PY" setup.py bdist_wheel --dist-dir "$ART/apex-wheel") || true
  local wheel
  wheel=$(ls -t "$ART"/apex-wheel/apex-*.whl 2>/dev/null | head -1)
  if [ -z "$wheel" ]; then
    echo "apex did not build: no wheel in $ART/apex-wheel" >&2
    return 1
  fi
  $UV pip install --python "$SFT_PY" "$wheel"
  "$SFT_PY" -c "import torch, amp_C" || { echo "apex installed but does not import" >&2; return 1; }
}

hf_model() {
  "$EVAL_VENV/bin/hf" download Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --local-dir "$SFT_MODEL_PATH"
}

hf_dataset() {
  # The evaluation environment's hf (huggingface_hub 0.36.2) reads every pattern after --include.
  "$EVAL_VENV/bin/hf" download Tianli/robovqa --repo-type dataset --revision eb5819490f0d9ec18a79774233d4b1c9ebd52eea \
    --include "videos/*" "json/*" "LICENSE.txt" --local-dir "$SRC"
}

download() {
  say "download: model and dataset at fixed revisions (no token needed)"
  stage "Qwen3-VL-8B-Instruct, 17 GB" download-model hf_model
  if [ "$SMOKE" = 1 ]; then
    printf '  %-62s%s\n' "RoboVQA" "skipped: --smoke uses the skill's six bundled clips"
  else
    stage "RoboVQA, 3.5 GB (9,999 clips)" download-dataset hf_dataset
  fi
  echo "  model on disk: $(du -sh "$SFT_MODEL_PATH" | cut -f1)   dataset: $(ls "$SRC/videos" | wc -l) clips"
}

data() {
  say "data: split, training frames, check (about 25 minutes on the full dataset)"
  stage "split 80/10/10 by clip, one question per clip" data-split \
    "$EVAL_PY" "$SKILL_DIR/scripts/prepare_robovqa.py" split --src "$SRC" --out "$DATA"
  stage "frames: each training clip's evaluation frames, written losslessly" data-frames \
    env CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH="$EVAL_LIBS" \
    "$EVAL_PY" "$SKILL_DIR/scripts/prepare_robovqa.py" frames --out "$DATA"
  stage "verify: the training stack sees exactly the evaluator's frames" data-verify \
    env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 LD_LIBRARY_PATH="$SFT_LIBS" \
    "$SFT_PY" "$SKILL_DIR/scripts/prepare_robovqa.py" verify --out "$DATA" --model "$SFT_MODEL_PATH"
  echo
  echo "  $(head -n 1 "$ART/logs/data-split$TAG.log")"
  echo "  frame check: $(tail -n 1 "$ART/logs/data-verify$TAG.log" | grep -o '"all_ok": [a-z]*, "bad": [0-9]*')"
}

sft_check() {  # sft_check <flag>: one of the SFT entrypoint's CPU checks
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. IMAGINAIRE_OUTPUT_ROOT="$RUN/preflight" LD_LIBRARY_PATH="$SFT_LIBS" \
    "$SFT_PY" -m cosmos_framework.posttrain.entrypoints.sft --config "$CONFIG" "$1"
}

preflight() {
  say "preflight: the recipe's CPU checks before any GPU time (about 13 minutes on the full dataset)"
  stage "config loads, roles wired (--validate-manifest)" preflight-manifest sft_check --validate-manifest
  stage "every training row packs (--validate-rows)" preflight-rows sft_check --validate-rows
  echo
  echo "  $(grep -m1 '^train: rows' "$ART/logs/preflight-rows$TAG.log")"
}

score() {  # score <model directory> <name>
  LD_LIBRARY_PATH="$EVAL_LIBS" "$EVAL_PY" "$SKILL_DIR/scripts/eval_robovqa.py" --model "$1" \
    --data "$DATA/hf/test.jsonl" --ref "$DATA/samples_test.jsonl" --out "$EVAL/$2"
}

before() {
  say "before: score the untouched model on the held-out clips (about 14 minutes for 1,000 clips)"
  stage "greedy answers, then exact match / token F1 / yes-no accuracy" before score "$SFT_MODEL_PATH" before
  echo
  echo "  $(tail -n 1 "$ART/logs/before$TAG.log")"
}

train() {
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -lt 150000 ]; then
    echo "${free} MiB free on the GPU; training the 8B model needs about 150 GB (measured peak 143,566 MiB)" >&2
    exit 1
  fi
  say "train: $STEPS updates of 64 clips on one GPU"
  echo "  the first 'step 1/$STEPS' line appears after about 3 minutes, then one line per update (about a minute each);"
  echo "  full log: $ART/logs/train$TAG.log"
  echo
  # Only the step lines and the end marker reach the screen; the framework's full log goes to the file.
  PYTHONPATH=. IMAGINAIRE_OUTPUT_ROOT="$RUN" LD_LIBRARY_PATH="$SFT_LIBS" \
    "$SFT_PY" -m torch.distributed.run --standalone --nproc_per_node=1 \
    -m cosmos_framework.posttrain.entrypoints.sft --config "$CONFIG" -- rl.max_steps="$STEPS" rl.save_every="$STEPS" \
    2>&1 | tee "$ART/logs/train$TAG.log" |
    grep --line-buffered -E ' step +[0-9]+/[0-9]+ |SFT DONE|Error|error:|Traceback' |
    sed -u -E 's/^.*(step +[0-9]+\/[0-9]+ +train\/loss_avg [+-]?[0-9.]+).*timer\/e2e_step_s ([+-]?[0-9.]+).*$/  \1   \2 s/; s/^.*SFT DONE/  SFT DONE/'
  grep -q "SFT DONE" "$ART/logs/train$TAG.log" || { echo "training did not finish; see $ART/logs/train$TAG.log" >&2; exit 1; }
}

materialize() {
  local ckpt
  ckpt=$(ls -d "$RUN"/posttrain_sft/qwen3_vl_8b_robovqa/seed0/*/checkpoint_artifacts/iter_* | sort | tail -1)
  echo "checkpoint: $ckpt"
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. LD_LIBRARY_PATH="$SFT_LIBS" \
    "$SFT_PY" -m cosmos_framework.posttrain.checkpoint.tools.materialize_hf \
    --ckpt-dir "$ckpt" --base "$SFT_MODEL_PATH" --out "$EXPORT" --expect-tensors 750 --min-tensors 750
}

export_hf() {
  say "export: the trained weights as a Hugging Face model (about 2 minutes, CPU)"
  stage "materialize the last checkpoint, verify all 750 tensors" export materialize
  echo
  echo "  $(grep -o '\[materialize\] OK.*' "$ART/logs/export$TAG.log")"
}

after() {
  say "after: score the fine-tuned model the same way"
  stage "greedy answers, then exact match / token F1 / yes-no accuracy" after score "$EXPORT" after
  echo
  echo "  $(tail -n 1 "$ART/logs/after$TAG.log")"
}

# Each stage writes its own log under $ART/logs/ (with a -smoke suffix for the smoke variant), so the steps
# themselves print only the short progress view.
case "$STEP" in
  setup | download | data | preflight | before | train | after) "$STEP" ;;
  export) export_hf ;;
  *)
    echo "usage: run.sh <setup|download|data|preflight|before|train|export|after> [--smoke]" >&2
    exit 2
    ;;
esac
