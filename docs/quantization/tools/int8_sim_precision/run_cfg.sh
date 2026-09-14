#!/bin/bash
# usage: run_cfg.sh <tag> [extra CLI args...]   (env vars QDQ_SIM_* are passed through)
set -e
TAG=$1; shift
cd /lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/cosmos-framework-int8
export HF_HOME=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/cosmos-framework-int8
rm -rf /lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench/sep_scale/run_$TAG
python -m cosmos_framework.scripts.inference --parallelism-preset=latency -i /lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench/sep_scale/t2i_set.jsonl -o /lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench/sep_scale/run_$TAG --checkpoint-path /lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/hf_cache/hub/models--nvidia--Cosmos3-Nano/snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa --no-guardrails --no-diffusion-cache "$@" 2>&1 | grep -E 'quantiz|Quantiz|matched|sha256|Saved sample outputs|Traceback|Error|rror:' | grep -v pkg_resources | tail -6
echo "== $TAG done: $(ls /lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench/sep_scale/run_$TAG/*/vision.jpg 2>/dev/null | wc -l) images; matched fqns: $(wc -l < /lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench/sep_scale/run_$TAG/quantization_matched_fqns.txt 2>/dev/null || echo n/a)"
[ "$TAG" != bf16 ] && python /lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench/sep_scale/psnr.py /lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench/sep_scale/run_bf16 /lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench/sep_scale/run_$TAG || true
