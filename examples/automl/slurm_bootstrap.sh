#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Runs inside the reviewed image. Stage/seal this file and both probe scripts;
# this is orchestration, not a replacement for code installed in the image.
set -Eeuo pipefail
: "${AUTOML_TRIAL_DIR:?executor supplies the trial directory}"
: "${AUTOML_GPUS:?executor supplies the allocated GPU count}"
: "${AUTOML_PREFLIGHT_PY:?path to the sealed Framework preflight script}"
: "${AUTOML_NCCL_PROBE_PY:?path to the sealed NCCL probe}"
: "${CUDA_VISIBLE_DEVICES:?scheduler must supply GPU visibility}"
workload_python="$1"
export LD_LIBRARY_PATH=''
export PATH="$(dirname "$workload_python"):$PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$AUTOML_TRIAL_DIR/cache/huggingface"
export XDG_CACHE_HOME="$AUTOML_TRIAL_DIR/cache"
export TRITON_CACHE_DIR="$AUTOML_TRIAL_DIR/cache/triton"
export TORCH_HOME="$AUTOML_TRIAL_DIR/cache/torch"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$AUTOML_TRIAL_DIR/preflight"
ulimit -n 65536
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv \
  > "$AUTOML_TRIAL_DIR/preflight/gpus.csv"
timeout --signal=TERM --kill-after=15 600 "$workload_python" "$AUTOML_PREFLIGHT_PY" \
  --trial-dir "$AUTOML_TRIAL_DIR" --gpus "$AUTOML_GPUS"
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
set +e
timeout --signal=TERM --kill-after=15 120 "$workload_python" \
  -m torch.distributed.run --standalone --nproc-per-node="$AUTOML_GPUS" \
  "$AUTOML_NCCL_PROBE_PY" > "$AUTOML_TRIAL_DIR/preflight/nccl-default.log" 2>&1
probe_exit=$?
set -e
if [ "$probe_exit" -eq 124 ] && [ "$NCCL_P2P_DISABLE" = 0 ]; then
  export NCCL_P2P_DISABLE=1
  timeout --signal=TERM --kill-after=15 120 "$workload_python" \
    -m torch.distributed.run --standalone --nproc-per-node="$AUTOML_GPUS" \
    "$AUTOML_NCCL_PROBE_PY" > "$AUTOML_TRIAL_DIR/preflight/nccl-p2p-disabled.log" 2>&1
  grep -q NCCL_PROBE_OK "$AUTOML_TRIAL_DIR/preflight/nccl-p2p-disabled.log"
elif [ "$probe_exit" -ne 0 ]; then
  exit "$probe_exit"
else
  grep -q NCCL_PROBE_OK "$AUTOML_TRIAL_DIR/preflight/nccl-default.log"
fi
printf 'NCCL_P2P_DISABLE=%s\n' "$NCCL_P2P_DISABLE" > "$AUTOML_TRIAL_DIR/preflight/nccl-selected.env"
exec "$@"
