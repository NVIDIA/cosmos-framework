#!/bin/bash
#SBATCH --account=cosmos_base_training
#SBATCH --job-name=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=interactive
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8-gemm/docs/quantization/tools/cutlass_g128_gemm/logs/slurm-%j.out
# FP8 g128 speed reference on GB200: stock CUTLASS example 81 groupwise (A 1x128; W 128x128 "blockwise" as shipped,
# and W 1x128 "per-col" via ScaleGranularityN=1). Timing = the example's own warm-L2 average over --iterations.
P=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren
D=$P/int8-gemm/docs/quantization/tools/cutlass_g128_gemm
C=/home/pzeren/cutlass
IMG=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/imaginaire4_v12.1.0.sqsh
srun --container-image=$IMG --container-mounts=/lustre:/lustre,/home/pzeren:/home/pzeren \
  bash -c "set -e; cd $D && export CUDA_VISIBLE_DEVICES=0 && B=build_ex81_$SLURM_JOB_ID && mkdir -p \$B && \
  cp $C/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_groupwise.cu \$B/ex81_bn128.cu && \
  sed 's/constexpr int ScaleGranularityN = 128;/constexpr int ScaleGranularityN = 1;/' \$B/ex81_bn128.cu > \$B/ex81_bn1.cu && grep -n 'ScaleGranularityN =' \$B/ex81_bn1.cu && \
  for v in bn128 bn1; do nvcc -std=c++17 -O3 -arch=sm_100a --expt-relaxed-constexpr -DCUTLASS_ENABLE_GDC_FOR_SM100=1 -I$C/include -I$C/tools/util/include -I$C/examples/common -lcuda -Xcompiler -Wno-psabi -diag-suppress 177 -o \$B/ex81_\$v \$B/ex81_\$v.cu & done; wait; ls -la \$B; \
  for v in bn128 bn1; do for nk in '4096 4096' '1024 4096' '12288 4096' '4096 12288'; do set -- \$nk; for m in 901 1802 4096 16384 42240; do \
     out=\$(timeout 120 \$B/ex81_\$v --m=\$m --n=\$1 --k=\$2 --iterations=50 2>&1 | grep -E 'Disposition|Avg runtime|GFLOPS' | tr '\n' ' '); echo \"EX81 fp8_g128_\$v N=\$1 K=\$2 M=\$m \$out\"; done; done; done"
