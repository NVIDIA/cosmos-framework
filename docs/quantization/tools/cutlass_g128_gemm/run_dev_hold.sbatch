#!/bin/bash
#SBATCH --account=cosmos_base_training
#SBATCH --job-name=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=interactive
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8-gemm/docs/quantization/tools/cutlass_g128_gemm/logs/hold-%j.out
# Long-lived allocation for iterative kernel development; work is attached with: srun --jobid=$SLURM_JOB_ID --overlap ...
echo "hold job $SLURM_JOB_ID on $(hostname) started $(date)"; sleep 10700
