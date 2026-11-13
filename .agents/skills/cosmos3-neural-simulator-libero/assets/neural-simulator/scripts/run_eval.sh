#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Full evaluation of one checkpoint: rollout + metrics, gripper check, report.
# Usage: scripts/run_eval.sh <iter e.g. 250> [run_dir (default runs/main/.../action_fd_libero_posttrain)] [extra fd_rollout_eval args...]
# Env overrides: EVAL_CKPT, EVAL_CFG, EVAL_OUT, EVAL_LABEL, EVAL_TITLE (e.g. control = base model, or seed repeats).
set -euo pipefail
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this directory: env.sh, timed.sh and the Python scripts live here
source "$SCRIPTS/env.sh"                                  # sets REPO, WORKDIR (=RECIPE), LIBERO_ROOT, LIBERO_HOLDOUT_PATH, ...
ITER=$1; shift
RUN_DIR=${1:-$RECIPE/runs/main/cosmos3_action_fd/action_sft/action_fd_libero_posttrain}; shift || true
CKPT=${EVAL_CKPT:-$RUN_DIR/checkpoints/iter_$(printf '%09d' "$ITER")}
CFG=${EVAL_CFG:-$RUN_DIR/config.yaml}
OUT=${EVAL_OUT:-$RECIPE/eval/iter$(printf '%04d' "$ITER")}
LABEL=${EVAL_LABEL:-iter${ITER}}
T=$SCRIPTS/timed.sh
[[ -d "$CKPT/model" ]] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "config not found: $CFG" >&2; exit 1; }
mkdir -p "$OUT" "$RECIPE/logs"
cd "$REPO"
PYTHONPATH=. $T "eval_${LABEL}_rollout" "$RECIPE/logs/eval_${LABEL}.log" \
  uv run --no-sync python "$SCRIPTS/fd_rollout_eval.py" --checkpoint-path "$CKPT" --config-file "$CFG" \
  --root "$LIBERO_ROOT" --holdout "$LIBERO_HOLDOUT_PATH" --out "$OUT" "$@"
if [[ -f "$RECIPE/eval/gripper/gripper_net.pt" ]]; then
  PYTHONPATH=. $T "eval_${LABEL}_gripper" "$RECIPE/logs/eval_${LABEL}.log" \
    uv run --no-sync python "$SCRIPTS/gripper_classifier.py" apply --model "$RECIPE/eval/gripper/gripper_net.pt" --eval-dir "$OUT"
else
  echo "gripper classifier not trained yet ($RECIPE/eval/gripper/gripper_net.pt missing); skipping gripper check" | tee -a "$RECIPE/logs/eval_${LABEL}.log"
fi
PYTHONPATH=. $T "eval_${LABEL}_report" "$RECIPE/logs/eval_${LABEL}.log" \
  uv run --no-sync python "$SCRIPTS/make_report.py" --eval-dir "$OUT" --title "LIBERO FD eval, ${EVAL_TITLE:-iteration $ITER}" --n-videos 10
echo "done: $OUT/summary.json  $OUT/report/index.html"
