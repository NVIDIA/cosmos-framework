#!/bin/bash
#SBATCH --account=cosmos_base_training
#SBATCH --job-name=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=interactive
#SBATCH --time=01:20:00
#SBATCH --output=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8-gemm/docs/quantization/tools/cutlass_pertensor_gemm/logs/slurm-%j.out

P=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren
D=$P/int8-gemm/docs/quantization/tools/cutlass_pertensor_gemm
IMG=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/imaginaire4_v12.1.0.sqsh
TAG=${TAG:-gb200_cluster_$(date +%F)_$SLURM_JOB_ID}
CFG="256x256x128_c2x2 256x128x128_c2x2 256x128x128_c4x1 256x256x128_c4x1 256x128x128_c2x1 256x256x128_c2x1"
srun --container-image=$IMG --container-mounts=/lustre:/lustre,/home/pzeren:/home/pzeren \
  bash -c "set -x; cd $D && export CUDA_VISIBLE_DEVICES=0 && \
  ls build_2159007 && \
  for b in build_2159007/pertensor_gemm_*; do timeout 60 \$b --m=301 --n=1024 --k=4096 --iterations=5 --verify_samples=4096 || echo VERIFY_FAILED \$b; done && \
  python3 bench.py --bin-dir build_2159007 --tag $TAG --iters 50 \
     --extra='' 2>&1 | tee logs/bench_$TAG.log"
