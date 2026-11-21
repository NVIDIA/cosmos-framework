#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Forward-dynamics evaluation of one checkpoint on the SutureBot val and test splits.
#
#   bash scripts/evaluate.sh <checkpoint model dir> <output dir>
#
# <checkpoint model dir> is a DCP "model" directory: the converted released checkpoint
# (<dcp>/model) or a training checkpoint (.../checkpoints/iter_<N>/model).
#
# Splits (same samples for every checkpoint):
#   val   held-out episodes of the training tissues (5% per tissue, seed 42)
#   test  the held-out tissue(s) (family suturebot_test: SUTUREBOT_TEST_TISSUES, default 3), every episode
#
# Env: OPENH_DATA_ROOT (required; holds suturebot/tissue_<N>/), NUM_SAMPLES (default 128), SEEDS (default "20260908"),
#      VAL_STRIDE / TEST_STRIDE (window stride between evaluated samples), NPROC (default 1),
#      FULL_METRICS=1 (also SSIM and LPIPS; LPIPS needs `pip install lpips`).

set -euo pipefail

CKPT=${1:?checkpoint model dir}
OUT=${2:?output dir}
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OPENH_DATA_ROOT:?set OPENH_DATA_ROOT}"
OPENH_DATA_ROOT="$(cd "$OPENH_DATA_ROOT" && pwd)"
export OPENH_DATA_ROOT  # SUTUREBOT_NORMALIZER, if set, must be the file the checkpoint was trained with
: "${NUM_SAMPLES:=128}"
: "${SEEDS:=20260908}"
: "${VAL_STRIDE:=24}"
: "${TEST_STRIDE:=34}"
: "${NPROC:=1}"

LAUNCH=(python)
[[ "$NPROC" == 1 ]] || LAUNCH=(torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-50013}")
EXTRA=()
[[ "${FULL_METRICS:-0}" == 1 ]] && EXTRA+=(--full-metrics)

cd "$RECIPE_DIR"
export PYTHONPATH="$RECIPE_DIR${PYTHONPATH:+:$PYTHONPATH}"

run() {  # split, data experiment, family, dataset split, stride, seed
    local out="$OUT/$1/seed_$6"
    if [[ -f "$out/metrics_aggregate.json" ]]; then
        echo "$out done, skipping"
        return
    fi
    "${LAUNCH[@]}" -m cosmos3_h_embodied.scripts.eval "${EXTRA[@]}" \
        --no-guardrails --checkpoint-path "$CKPT" \
        --config-file suturebot_fd/config.py --experiment suturebot_fd \
        --dataset.dataset "$3" --dataset.dataset-experiment "$2" --dataset.dataset-split "$4" \
        --dataset.model-mode forward_dynamics --dataset.num-samples "$NUM_SAMPLES" --dataset.sample-stride "$5" \
        --seed "$6" --num-steps 35 --guidance 6.0 -o "$out"
}

for seed in $SEEDS; do
    run val suturebot_fd suturebot val "$VAL_STRIDE" "$seed"
    run test suturebot_fd_test suturebot_test full "$TEST_STRIDE" "$seed"
done
