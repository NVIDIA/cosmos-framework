#!/usr/bin/env bash
# Source this to activate the lance venv with the CUDA/NPP/ffmpeg libs torchcodec needs.
# Usage:  source benchmarks/lance/_env.sh
VENV=/home/ubuntu/.venv-lance
SP=$VENV/lib/python3.12/site-packages
NVLIB=$(ls -d $SP/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIB}${SP}/torch/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="$VENV/bin:${PATH:-}"
export PYTHONPATH="/home/ubuntu/work/cosmos-framework:${PYTHONPATH:-}"
