#!/bin/bash
#SBATCH --account=cosmos_base_training
#SBATCH --job-name=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=interactive
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8-gemm/docs/quantization/tools/cutlass_g128_gemm/logs/slurm-%j.out
# Usage: KINDS="fp8 int8" CONFIGS="256x128x128_c2x1_sfn1 ..." MS="901,1802,4096,16384,42240" sbatch run_g128.sbatch
P=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren
D=$P/int8-gemm/docs/quantization/tools/cutlass_g128_gemm
IMG=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/imaginaire4_v12.1.0.sqsh
KINDS=${KINDS:-fp8}
CONFIGS=${CONFIGS:-256x128x128_c2x1_sfn1 256x128x128_c2x1_sfn128 128x128x128_c1x1_sfn1 256x256x128_c2x1_sfn1}
MS=${MS:-901,1802,4096,16384,42240}
TAG=${TAG:-g128_$(date +%F)_$SLURM_JOB_ID}
srun --container-image=$IMG --container-mounts=/lustre:/lustre,/home/pzeren:/home/pzeren \
  bash -c "set -x; cd $D && export CUDA_VISIBLE_DEVICES=0 && B=build_$SLURM_JOB_ID && \
  (time make -j10 CUTLASS=/home/pzeren/cutlass OUT=\$B KINDS='$KINDS' CONFIGS='$CONFIGS') 2>&1 | grep -v '^nvcc' | tail -40; ls \$B || exit 1; \
  for b in \$B/g128_gemm_*; do timeout 120 \$b --m=301 --n=1024 --k=4096 --iterations=5 --verify_samples=4096 || echo VERIFY_FAILED \$b; done; \
  for b in \$B/g128_gemm_*; do for nk in '4096 4096' '1024 4096' '12288 4096' '4096 12288'; do set -- \$nk; for m in \$(echo $MS | tr , ' '); do \
     timeout 180 \$b --m=\$m --n=\$1 --k=\$2 --iterations=50 --verify_samples=1024 || echo RUN_FAILED \$b m=\$m n=\$1 k=\$2; done; done; done 2>&1 | tee logs/bench_$TAG.log"
