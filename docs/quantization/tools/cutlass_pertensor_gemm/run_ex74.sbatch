#!/bin/bash
#SBATCH --account=cosmos_base_training
#SBATCH --job-name=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=interactive
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8-gemm/docs/quantization/tools/cutlass_pertensor_gemm/logs/slurm-%j.out

P=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren
D=$P/int8-gemm/docs/quantization/tools/cutlass_pertensor_gemm
C=/home/pzeren/cutlass
IMG=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/imaginaire4_v12.1.0.sqsh
srun --container-image=$IMG --container-mounts=/lustre:/lustre,/home/pzeren:/home/pzeren \
  bash -c "set -x; cd $D && export CUDA_VISIBLE_DEVICES=0 && mkdir -p build_ex74_$SLURM_JOB_ID && \
  (time nvcc -std=c++17 -O3 -arch=sm_100a --expt-relaxed-constexpr -DCUTLASS_ENABLE_GDC_FOR_SM100=1 -I$C/include -I$C/tools/util/include -I$C/examples/common -lcuda -Xcompiler -Wno-psabi -diag-suppress 177 \
     -o build_ex74_$SLURM_JOB_ID/ex74 $C/examples/74_blackwell_gemm_streamk/blackwell_gemm_streamk.cu) 2>&1 | tail -5; \
  for args in '--m=256 --n=256 --k=16384 --decomposition=Heuristic' '--m=301 --n=1024 --k=4096 --decomposition=Heuristic' '--m=301 --n=1024 --k=4096 --decomposition=DataParallel' '--m=301 --n=1024 --k=4096 --decomposition=StreamK' '--m=301 --n=1024 --k=4096 --decomposition=SplitK --splits=4' '--m=901 --n=1024 --k=4096 --decomposition=StreamK'; do \
     echo \"== ex74 \$args\"; timeout 60 build_ex74_$SLURM_JOB_ID/ex74 \$args --iterations=20 || echo EX74_FAILED_OR_TIMEOUT; done; \
  echo '=== own kernel, unit-test config 64x128x128 c1x4 sk (int8) ==='; \
  (time make -j2 CUTLASS=$C ARCH=100 OUT=build_ex74_$SLURM_JOB_ID CONFIGS='64x128x128_c1x4_sk 128x128x128_c1x1_sk' KINDS='int8') 2>&1 | grep -v '^nvcc' | tail -5; \
  for b in build_ex74_$SLURM_JOB_ID/pertensor_gemm_*; do for d in DataParallel Heuristic; do echo \"== \$b \$d\"; timeout 60 \$b --m=301 --n=1024 --k=4096 --iterations=5 --verify_samples=1024 --decomposition=\$d || echo OWN_FAILED_OR_TIMEOUT; done; done"
