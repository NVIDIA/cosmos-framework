#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Paths and tool pins shared by every SKILL.md block. Source it from the repository root at the top of each block:
#   source .agents/skills/cosmos3-qwen3-vl-robovqa-sft/scripts/env.sh
# It only sets variables (and creates $ART); it installs nothing.
if [ ! -f pyproject.toml ] || [ ! -d cosmos_framework ]; then
  echo "env.sh: run from the cosmos-framework repository root (the directory with pyproject.toml)" >&2
  return 1 2>/dev/null || exit 1
fi
export PATH="$HOME/.local/bin:$PATH"                     # where the uv installer puts uv / uvx
export SKILL_DIR="$PWD/.agents/skills/cosmos3-qwen3-vl-robovqa-sft"
export UV="uvx --from uv==0.11.29 uv"                    # the checkout pins uv 0.11.29 (tool.uv.required-version)
export ART="$PWD/artifacts/qwen3-vl-robovqa"             # every artifact of this skill (about 110 GB for the full run)
export SFT_VENV="$PWD/docker/posttrain/environment/.venv-sft"
export SFT_PY="$SFT_VENV/bin/python"
export SFT_LIBS="$SFT_VENV/lib/python3.13/site-packages/nvidia/cu13/lib"    # libnppicc.so.13 for TorchCodec 0.11
export EVAL_VENV="$ART/venvs/eval"
export EVAL_PY="$EVAL_VENV/bin/python"
export EVAL_LIBS="$EVAL_VENV/lib/python3.12/site-packages/nvidia/cu13/lib"  # libnppicc.so.13 for TorchCodec 0.10
export SFT_MODEL_PATH="$ART/models/Qwen3-VL-8B-Instruct"  # read by the recipe config
export CONFIG="cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_robovqa.py"
mkdir -p "$ART/logs"
