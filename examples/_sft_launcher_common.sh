# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Shared launch plumbing for examples/launch_sft_<recipe>.sh — the
# structured-TOML / pydantic-schema flow that drives cosmos_framework.scripts.train.
#
# Caller MUST set before sourcing:
#   TOML_FILE            recipe TOML, e.g. "examples/toml/sft_config/<recipe>.toml".
#                        Absolute or repo-root-relative.
#
# Caller MAY set before sourcing (presence drives which existence checks fire):
#   DATASET_PATH         recipe-local dataset dir, e.g. "examples/data/<name>".
#                        If unset, no dataset existence check fires
#                        (reasoner / HF-streaming case).
#   BASE_CHECKPOINT_PATH recipe-local base DCP dir, e.g. "examples/checkpoints/<name>".
#                        Setting it also enables WAN_VAE_PATH plumbing + check.
#   WAN_VAE_PATH         override the default examples/checkpoints/wan22_vae/Wan2.2_VAE.pth.
#   EXTRA_DATASET_CHECK  bash snippet (string) eval'd after the default checks.
#   TAIL_OVERRIDES       bash array of Hydra CLI overrides appended after `--`
#                        (e.g. data_setting.max_tokens=16000 for VLM smokes).
#   MASTER_PORT          torchrun --master_port; default 50012.
#   NPROC_PER_NODE       torchrun --nproc_per_node; default 8.
#   NNODES               torchrun --nnodes; multi-node only (unset = single-node).
#   NODE_RANK            torchrun --node_rank; this worker's 0-based index.
#   MASTER_ADDR          torchrun --master_addr; rank-0 host (multi-node only — it
#                        has no torchrun env fallback, so it must be passed here).
#   LOG_FILENAME         override $LOG_DIR/${LOG_FILENAME}
#                        (default <toml-stem>_sft.log).
#
# Absolute paths are passed through; relative paths are anchored to the repo
# root (the parent of this examples/ directory). Paths set in the caller's
# shell via `export DATASET_PATH=...` etc. win over the launcher's defaults
# (use the `: "${VAR:=default}"` idiom in the launcher to preserve this).

set -uo pipefail

: "${TOML_FILE:?TOML_FILE must be set before sourcing _sft_launcher_common.sh}"

# Repo root = parent of the wrapper's directory (examples/).
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"

# Anchor relative paths to $WORKDIR.
[[ "$TOML_FILE" = /* ]] || TOML_FILE="$WORKDIR/$TOML_FILE"

if [[ -n "${DATASET_PATH:-}" ]]; then
    [[ "$DATASET_PATH" = /* ]] || DATASET_PATH="$WORKDIR/$DATASET_PATH"
    export DATASET_PATH
fi

if [[ -n "${BASE_CHECKPOINT_PATH:-}" ]]; then
    [[ "$BASE_CHECKPOINT_PATH" = /* ]] || BASE_CHECKPOINT_PATH="$WORKDIR/$BASE_CHECKPOINT_PATH"
    WAN_VAE_PATH="${WAN_VAE_PATH:-examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
    [[ "$WAN_VAE_PATH" = /* ]] || WAN_VAE_PATH="$WORKDIR/$WAN_VAE_PATH"
    export BASE_CHECKPOINT_PATH WAN_VAE_PATH
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKDIR/outputs/train}"
LOG_DIR="$OUTPUT_ROOT/logs"
TOML_STEM="$(basename "$TOML_FILE" .toml)"
LOG_FILE="$LOG_DIR/${LOG_FILENAME:-${TOML_STEM}_sft.log}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
mkdir -p "$LOG_DIR"

echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -f "$TOML_FILE" ]] || { echo "ERROR: TOML not found: $TOML_FILE" >&2; exit 1; }
if [[ -n "${DATASET_PATH:-}" ]]; then
    [[ -d "$DATASET_PATH" ]] || { echo "ERROR: DATASET_PATH not found: $DATASET_PATH (run Step 1 of docs/training.md, or export DATASET_PATH=<path>)" >&2; exit 1; }
fi
if [[ -n "${BASE_CHECKPOINT_PATH:-}" ]]; then
    [[ -d "$BASE_CHECKPOINT_PATH" ]] || { echo "ERROR: BASE_CHECKPOINT_PATH not found: $BASE_CHECKPOINT_PATH (run Step 2 of docs/training.md, or export BASE_CHECKPOINT_PATH=<path>)" >&2; exit 1; }
    [[ -f "$WAN_VAE_PATH" ]]         || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH (run Step 1 of docs/training.md, or export WAN_VAE_PATH=<path>)" >&2; exit 1; }
fi
if [[ -n "${EXTRA_DATASET_CHECK:-}" ]]; then eval "$EXTRA_DATASET_CHECK"; fi

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') WORKDIR:    $WORKDIR"
echo ">>> $(date '+%H:%M:%S') TOML:       $TOML_FILE"
[[ -n "${DATASET_PATH:-}" ]]         && echo ">>> $(date '+%H:%M:%S') dataset:    $DATASET_PATH"
[[ -n "${BASE_CHECKPOINT_PATH:-}" ]] && echo ">>> $(date '+%H:%M:%S') checkpoint: $BASE_CHECKPOINT_PATH"
echo ">>> $(date '+%H:%M:%S') log:        $LOG_FILE"

# Default empty if caller didn't set; safe under set -u.
[[ ${TAIL_OVERRIDES+x} ]] || TAIL_OVERRIDES=()

TRAILING_ARGS=()
if (( ${#TAIL_OVERRIDES[@]} > 0 )); then
    TRAILING_ARGS=(-- "${TAIL_OVERRIDES[@]}")
fi

# torchrun topology. Single-node by default; a SLURM/Lepton wrapper sets NNODES /
# NODE_RANK / MASTER_ADDR for multi-node. Each is appended only when set, so with all
# three unset the invocation is identical to the single-node case.
TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE:-8}" --master_port="${MASTER_PORT:-50012}")
[[ -n "${NNODES:-}" ]]      && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]]   && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

IMAGINAIRE_OUTPUT_ROOT="$IMAGINAIRE_OUTPUT_ROOT" PYTHONPATH=. \
    torchrun "${TORCHRUN_ARGS[@]}" -m cosmos_framework.scripts.train \
    --sft-toml="$TOML_FILE" \
    "${TRAILING_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
