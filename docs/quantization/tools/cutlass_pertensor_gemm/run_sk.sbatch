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
TAG=${TAG:-gb200_streamk_$(date +%F)_$SLURM_JOB_ID}
SK="256x128x128_c2x1_sk 128x128x128_c1x1_sk 256x256x128_c2x1_sk"
SM90PP="64x128x128_c1x2_pp 64x256x128_c1x2_pp"
srun --container-image=$IMG --container-mounts=/lustre:/lustre,/home/pzeren:/home/pzeren \
  bash -c "set -x; cd $D && export CUDA_VISIBLE_DEVICES=0 && \
  (time make -j10 CUTLASS=/home/pzeren/cutlass ARCH=100 OUT=build_$SLURM_JOB_ID CONFIGS='$SK') 2>&1 | grep -v '^nvcc' | tail -20 && ls build_$SLURM_JOB_ID && \
  for b in build_$SLURM_JOB_ID/pertensor_gemm_*; do for d in Heuristic StreamK 'SplitK --splits=4' DataParallel; do timeout 60 \$b --m=301 --n=1024 --k=4096 --iterations=5 --verify_samples=4096 --decomposition=\$d || echo VERIFY_FAILED \$b \$d; done; done && \
  python3 bench.py --bin-dir build_$SLURM_JOB_ID --tag $TAG --ms 901,1802,4096 --iters 50 --skip-baselines \
     --extra="--decomposition=Heuristic" --extra="--decomposition=StreamK" --extra="--decomposition=SplitK --splits=2" --extra="--decomposition=SplitK --splits=4" --extra="--decomposition=DataParallel" 2>&1 | tee logs/bench_$TAG.log; \
  echo '=== SM90 pingpong compile check ==='; \
  (time make -j4 CUTLASS=/home/pzeren/cutlass ARCH=90 OUT=build_sm90_$SLURM_JOB_ID CONFIGS='$SM90PP') 2>&1 | grep -v '^nvcc' | tail -20; ls -la build_sm90_$SLURM_JOB_ID"
