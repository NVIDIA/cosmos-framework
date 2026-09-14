#!/bin/bash
#SBATCH --account=cosmos_base_training
#SBATCH --job-name=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=interactive
#SBATCH --time=00:25:00
#SBATCH --output=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8-gemm/docs/quantization/tools/cutlass_g128_gemm/logs/slurm-%j.out
# Compile-only check that both tools build for Thor (sm_110a) and still for GB200 (sm_100a). No Thor GPU here to run them.
T=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8-gemm/docs/quantization/tools
IMG=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/imaginaire4_v12.1.0.sqsh
srun --container-image=$IMG --container-mounts=/lustre:/lustre,/home/pzeren:/home/pzeren bash -c "set -x; nvcc --list-gpu-arch | tr '\n' ' '; echo; \
  cd $T/cutlass_g128_gemm && (time make -j8 CUTLASS=/home/pzeren/cutlass ARCH=110 OUT=build_sm110_$SLURM_JOB_ID KINDS='int8 fp8' CONFIGS='256x128x256_c2x1_sfn1 256x128x128_c2x1_sfn128' EXTRA='-DG128_OPT_PROMOTION -DG128_OPT_PIPELINE') 2>&1 | grep -v '^nvcc' | grep -v 'warning #' | tail -15; ls -la build_sm110_$SLURM_JOB_ID; \
  cd $T/cutlass_pertensor_gemm && (time make -j8 CUTLASS=/home/pzeren/cutlass ARCH=110 OUT=build_sm110_$SLURM_JOB_ID KINDS='int8 fp8' CONFIGS='256x256x128_c2x1 256x128x128_c2x1') 2>&1 | grep -v '^nvcc' | grep -v 'warning #' | tail -15; ls -la build_sm110_$SLURM_JOB_ID; \
  echo '=== sm_100a regression build + run of the edited sources ==='; export CUDA_VISIBLE_DEVICES=0; \
  (make -j8 CUTLASS=/home/pzeren/cutlass ARCH=100 OUT=build_sm100_$SLURM_JOB_ID KINDS=int8 CONFIGS='256x128x128_c2x1') 2>&1 | grep -v '^nvcc' | tail -3; build_sm100_$SLURM_JOB_ID/pertensor_gemm_int8_256x128x128_c2x1 --m=901 --n=4096 --k=4096 --iterations=20; \
  cd $T/cutlass_g128_gemm && (make -j8 CUTLASS=/home/pzeren/cutlass ARCH=100 OUT=build_sm100_$SLURM_JOB_ID KINDS=int8 CONFIGS='256x128x256_c2x1_sfn1' EXTRA='-DG128_OPT_PROMOTION -DG128_OPT_PIPELINE') 2>&1 | grep -v '^nvcc' | tail -3; build_sm100_$SLURM_JOB_ID/g128_gemm_int8_256x128x256_c2x1_sfn1 --m=901 --n=4096 --k=4096 --iterations=20"
