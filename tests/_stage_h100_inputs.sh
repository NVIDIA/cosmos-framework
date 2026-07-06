#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# One-shot helper: bootstraps the venv, stages the inputs the H100 regression
# run needs into a persistent lustre directory, and writes an `env.sh` that
# `source`s the venv + all paths needed by tests/launch_regression_test.py.
# Intended to run inside the i4 training container (torch.25.05.sqsh).
#
# Usage (from any directory, inside the container):
#   export HF_TOKEN=hf_...
#   bash tests/_stage_h100_inputs.sh
#   source $STAGE_DIR/env.sh        # default STAGE_DIR is below
#   pytest -s tests/launch_regression_test.py::test_launch_regression \
#       --num-gpus=4 --levels=2 -o addopts=
#
# Idempotent: re-running skips HF downloads already cached and the DCP convert
# when the output dir exists. Override STAGE_DIR / REPO_ROOT / UV_GROUP via env.
set -uo pipefail

: "${STAGE_DIR:=/path/to/cosmos_assets}"
: "${REPO_ROOT:=/path/to/Cosmos}"
UV_GROUP="${UV_GROUP:-cu128-train}"           # cu130-train for cuda 13.0 containers

mkdir -p "$STAGE_DIR"

export HF_HOME="${HF_HOME:-$STAGE_DIR/hf_cache}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME"

: "${HF_TOKEN:?HF_TOKEN required (export your Hugging Face token before running)}"

echo ">>> $(date '+%H:%M:%S') HF_HOME=$HF_HOME STAGE_DIR=$STAGE_DIR REPO_ROOT=$REPO_ROOT"

# ----------------------------------------------------------------------------
# 0. Python env: uv sync + pinned transformers. cosmos_framework/utils/generator/monkey_patch.py
#    hard-rejects every transformers version except 4.57.1 (pyproject's
#    `>=4.57.1,<5.0` is looser than what actually works at runtime).
# ----------------------------------------------------------------------------
echo ">>> $(date '+%H:%M:%S') uv sync ($UV_GROUP) at $REPO_ROOT ..."
cd "$REPO_ROOT"
uv sync --all-extras --group="$UV_GROUP"
# shellcheck disable=SC1091
source .venv/bin/activate
export LD_LIBRARY_PATH=""

if [[ "$(python -c 'import transformers; print(transformers.__version__)')" != "4.57.1" ]]; then
    echo ">>> $(date '+%H:%M:%S') pinning transformers==4.57.1 ..."
    uv pip install 'transformers==4.57.1' >/dev/null
fi
echo ">>> $(date '+%H:%M:%S') transformers=$(python -c 'import transformers; print(transformers.__version__)')"

# ----------------------------------------------------------------------------
# 1. Mixed-modality SFT dataset (BridgeData2-Subset-Synthetic-Captions).
# ----------------------------------------------------------------------------
echo ">>> $(date '+%H:%M:%S') downloading BridgeData2-Subset-Synthetic-Captions ..."
BRIDGE_ROOT=$(uvx hf@latest download --repo-type dataset \
    nvidia/BridgeData2-Subset-Synthetic-Captions \
    --revision 40d018ac1c1a2a4b9734f17fdb21f3d933c49a01 \
    --quiet)
DATASET_PATH="$BRIDGE_ROOT/sft_dataset_bridge"
echo "DATASET_PATH=$DATASET_PATH"

# ----------------------------------------------------------------------------
# 2. Wan2.2 VAE checkpoint.
# ----------------------------------------------------------------------------
echo ">>> $(date '+%H:%M:%S') downloading Wan2.2_VAE.pth ..."
WAN_VAE_PATH=$(uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --quiet)
echo "WAN_VAE_PATH=$WAN_VAE_PATH"

# ----------------------------------------------------------------------------
# 3. VLM backbone for launch_vlm_llava_ov (Qwen3-VL-8B-Instruct). Cosmos's
#    tokenizer dispatcher checks for the substring `Qwen/Qwen3-VL` in the path
#    (cosmos_framework/data/generator/processors/__init__.py); HF's cache uses
#    `models--Qwen--Qwen3-VL-8B-Instruct/snapshots/...` which doesn't match. We
#    add a `$STAGE_DIR/Qwen/Qwen3-VL-8B-Instruct` symlink so the dispatched
#    substring is present, and point `MODEL_PATH` at the symlink.
# ----------------------------------------------------------------------------
echo ">>> $(date '+%H:%M:%S') downloading Qwen3-VL-8B-Instruct ..."
_HF_SNAP=$(uvx hf@latest download Qwen/Qwen3-VL-8B-Instruct \
    --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --quiet)
mkdir -p "$STAGE_DIR/Qwen"
ln -sfn "$_HF_SNAP" "$STAGE_DIR/Qwen/Qwen3-VL-8B-Instruct"
MODEL_PATH="$STAGE_DIR/Qwen/Qwen3-VL-8B-Instruct"
echo "MODEL_PATH=$MODEL_PATH"

# ----------------------------------------------------------------------------
# 4. Base DCP checkpoint converted from Cosmos3-Nano (registered HF repo).
# ----------------------------------------------------------------------------
BASE_CHECKPOINT_PATH="$STAGE_DIR/Cosmos3-Nano-DCP"
if [[ ! -d "$BASE_CHECKPOINT_PATH/model" ]]; then
    echo ">>> $(date '+%H:%M:%S') converting Cosmos3-Nano -> DCP at $BASE_CHECKPOINT_PATH ..."
    PYTHONPATH=. python -m cosmos_framework.scripts.convert_model_to_dcp \
        -o "$BASE_CHECKPOINT_PATH" \
        --checkpoint-path Cosmos3-Nano
else
    echo ">>> $(date '+%H:%M:%S') $BASE_CHECKPOINT_PATH already populated; skipping conversion"
fi
echo "BASE_CHECKPOINT_PATH=$BASE_CHECKPOINT_PATH"

# ----------------------------------------------------------------------------
# Emit env.sh. Sourcing it should be sufficient prep to run pytest from a
# fresh shell: it activates the venv, sets the cosmos_framework-required env tweaks,
# and exports the four input paths the test reads.
# ----------------------------------------------------------------------------
cat > "$STAGE_DIR/env.sh" <<EOF
# Auto-generated by tests/_stage_h100_inputs.sh — sourcing this leaves the
# shell ready to run \`pytest -s tests/launch_regression_test.py ...\`.

# Python env
source "$REPO_ROOT/.venv/bin/activate"
export LD_LIBRARY_PATH=""

# Hugging Face cache (downloads land here)
export HF_HOME="$HF_HOME"
export HF_HUB_DISABLE_XET=1

# Regression test inputs
export DATASET_PATH="$DATASET_PATH"
export WAN_VAE_PATH="$WAN_VAE_PATH"
export MODEL_PATH="$MODEL_PATH"
export BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH"

# pytest needs this to accept --num-gpus=4 (see cosmos_framework/inference/fixtures/args.py)
export TEST_MAX_GPUS=4
EOF
echo ">>> $(date '+%H:%M:%S') wrote $STAGE_DIR/env.sh"
echo ">>> $(date '+%H:%M:%S') done. To run the regression now:"
echo "    source $STAGE_DIR/env.sh"
echo "    pytest -s tests/launch_regression_test.py::test_launch_regression \\"
echo "        --num-gpus=4 --levels=2 -o addopts="
