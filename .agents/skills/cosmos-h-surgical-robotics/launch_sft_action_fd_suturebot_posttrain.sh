#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Forward-dynamics post-training of Cosmos3-H-Embodied-Nano on SutureBot.
#
# The experiment lives in suturebot_fd/config.py (a Hydra experiment on top of
# cosmos3_h_embodied; cosmos-framework's --sft-toml loader cannot load plugin
# experiments). This launcher exports the resolved config to YAML, then starts
# cosmos-framework's trainer on it. Run it from anywhere, in the
# cosmos-framework environment with cosmos3_h_embodied installed.
#
# Env vars:
#   OPENH_DATA_ROOT          data directory holding the converted roots suturebot/tissue_<N>/ (required)
#   BASE_CHECKPOINT_PATH     DCP directory of the released checkpoint (required unless SMOKE=1)
#   IMAGINAIRE_OUTPUT_ROOT   output root (default: <recipe>/outputs/train)
#   SUTUREBOT_TRAIN_TISSUES  comma-separated tissue IDs (default 1,2,4,5,6,7,8,9,10)
#   SUTUREBOT_NORMALIZER     action statistics JSON (default: the SutureBot file shipped with cosmos3_h_embodied)
#   EXPERIMENT               suturebot_fd (default) or suturebot_fd_smoke
#   SMOKE=1                  1 GPU, 2-layer model (COSMOS_SMOKE=1), suturebot_fd_smoke, no base checkpoint
#   NPROC_PER_NODE           GPUs per node (default 8)
#   NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT   multi-node (torchrun)
#   EXTRA_OVERRIDES          space-separated Hydra overrides, e.g. "trainer.max_iter=500"
#   WANDB_API_KEY            with EXTRA_OVERRIDES="job.wandb_mode=online"
#
# Examples:
#   SMOKE=1 bash launch_sft_action_fd_suturebot_posttrain.sh
#   BASE_CHECKPOINT_PATH=/ckpt/Cosmos3-H-Embodied-Nano_dcp bash launch_sft_action_fd_suturebot_posttrain.sh
# ============================================================================

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OPENH_DATA_ROOT:?set OPENH_DATA_ROOT to the data directory holding suturebot/tissue_<N>/}"
: "${SMOKE:=0}"
: "${IMAGINAIRE_OUTPUT_ROOT:=$RECIPE_DIR/outputs/train}"
OPENH_DATA_ROOT="$(cd "$OPENH_DATA_ROOT" && pwd)"
export OPENH_DATA_ROOT IMAGINAIRE_OUTPUT_ROOT

OVERRIDES=()
if [[ "$SMOKE" == 1 ]]; then
    : "${EXPERIMENT:=suturebot_fd_smoke}"
    NPROC_PER_NODE=1
    export COSMOS_SMOKE=1
    OVERRIDES+=(model.config.compile.enabled=false)
else
    : "${EXPERIMENT:=suturebot_fd}"
    : "${BASE_CHECKPOINT_PATH:?set BASE_CHECKPOINT_PATH to the DCP directory of the released checkpoint}"
    [[ -d "$BASE_CHECKPOINT_PATH/model" ]] || { echo "ERROR: $BASE_CHECKPOINT_PATH/model not found (convert the released checkpoint first)" >&2; exit 1; }
    OVERRIDES+=(checkpoint.load_path="$BASE_CHECKPOINT_PATH")
fi
: "${NPROC_PER_NODE:=8}"
WORLD_SIZE=$(( NPROC_PER_NODE * ${NNODES:-1} ))
OVERRIDES+=(model.config.parallelism.data_parallel_shard_degree="$WORLD_SIZE")
# shellcheck disable=SC2206
OVERRIDES+=(${EXTRA_OVERRIDES:-})

if [[ -n "${SUTUREBOT_NORMALIZER:-}" ]]; then
    [[ -f "$SUTUREBOT_NORMALIZER" ]] || { echo "ERROR: SUTUREBOT_NORMALIZER=$SUTUREBOT_NORMALIZER not found" >&2; exit 1; }
    SUTUREBOT_NORMALIZER="$(cd "$(dirname "$SUTUREBOT_NORMALIZER")" && pwd)/$(basename "$SUTUREBOT_NORMALIZER")"
    export SUTUREBOT_NORMALIZER
fi
for t in $(echo "${SUTUREBOT_TRAIN_TISSUES:-1,2,4,5,6,7,8,9,10}" | tr ',' ' '); do
    [[ -f "$OPENH_DATA_ROOT/suturebot/tissue_$t/meta/modality.json" ]] || { echo "ERROR: $OPENH_DATA_ROOT/suturebot/tissue_$t is not a converted root" >&2; exit 1; }
done

cd "$RECIPE_DIR"
export PYTHONPATH="$RECIPE_DIR${PYTHONPATH:+:$PYTHONPATH}"
LOG_DIR="$IMAGINAIRE_OUTPUT_ROOT/logs"
CONFIG_YAML="$IMAGINAIRE_OUTPUT_ROOT/configs/$EXPERIMENT.yaml"
mkdir -p "$LOG_DIR" "$(dirname "$CONFIG_YAML")"

echo ">>> $(date '+%H:%M:%S') experiment $EXPERIMENT on $WORLD_SIZE GPU(s); data $OPENH_DATA_ROOT/suturebot"
python -m cosmos_framework.scripts.export_config -o "$CONFIG_YAML" \
    --config-file suturebot_fd/config.py --experiment "$EXPERIMENT"

TORCHRUN_ARGS=(--nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-50012}")
[[ -n "${NNODES:-}" ]] && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]] && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

set +e
torchrun "${TORCHRUN_ARGS[@]}" -m cosmos_framework.scripts._train \
    --config-file "$CONFIG_YAML" -o "$IMAGINAIRE_OUTPUT_ROOT/runs/$EXPERIMENT" \
    --config-overrides "${OVERRIDES[@]}" \
    2>&1 | tee "$LOG_DIR/$EXPERIMENT.log"
EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') done (exit $EXIT_CODE); checkpoints: $IMAGINAIRE_OUTPUT_ROOT/cosmos3_h_embodied/openh_h16/$EXPERIMENT/checkpoints"
exit "$EXIT_CODE"
