#!/bin/bash
#SBATCH --account=cosmos_base_training
#SBATCH --job-name=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=interactive
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8-gemm/docs/quantization/tools/cutlass_pertensor_gemm/logs/slurm-%j.out

P=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren
D=$P/int8-gemm/docs/quantization/tools/cutlass_pertensor_gemm
IMG=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/imaginaire4_v12.1.0.sqsh
TAG=${TAG:-gb200_smallM_$(date +%F)_$SLURM_JOB_ID}
SMALL="64x128x128_c1x1 128x64x128_c1x1 64x64x128_c1x1 128x128x128_c1x1_sk 256x128x128_c2x1_sk"
SM90="128x128x128_c1x2 128x256x128_c1x2 128x128x128_c2x1"
srun --container-image=$IMG --container-mounts=/lustre:/lustre,/home/pzeren:/home/pzeren \
  bash -c "set -x; cd $D && export CUDA_VISIBLE_DEVICES=0 && \
  (time make -j10 CUTLASS=/home/pzeren/cutlass ARCH=100 OUT=build_$SLURM_JOB_ID CONFIGS='$SMALL') 2>&1 | grep -v '^nvcc' | tail -20 && ls build_$SLURM_JOB_ID && \
  for b in build_$SLURM_JOB_ID/pertensor_gemm_*; do \$b --m=301 --n=1024 --k=4096 --iterations=5 --verify_samples=4096 || echo VERIFY_FAILED \$b; done && \
  python3 bench.py --bin-dir build_$SLURM_JOB_ID --tag $TAG --ms 901,1802,4096 --iters 50 2>&1 | tee logs/bench_$TAG.log; \
  echo '=== SM90 compile check (not run: no H100 here) ==='; \
  (time make -j6 CUTLASS=/home/pzeren/cutlass ARCH=90 OUT=build_sm90_$SLURM_JOB_ID CONFIGS='$SM90') 2>&1 | grep -v '^nvcc' | tail -20; ls -la build_sm90_$SLURM_JOB_ID"
