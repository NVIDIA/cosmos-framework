#!/bin/bash
#SBATCH --account=cosmos_base_training
#SBATCH --job-name=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=interactive
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8-gemm/docs/quantization/tools/cutlass_pertensor_gemm/logs/slurm-%j.out

P=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren
D=$P/int8-gemm/docs/quantization/tools/cutlass_pertensor_gemm
IMG=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/containers/imaginaire4_v12.1.0.sqsh
B=build_2158916
srun --container-image=$IMG --container-mounts=/lustre:/lustre,/home/pzeren:/home/pzeren \
  bash -c "cd $D && export CUDA_VISIBLE_DEVICES=0 && nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv && \
  for shp in '--m=8192 --n=8192 --k=8192' '--m=16384 --n=8192 --k=8192' '--m=16384 --n=16384 --k=8192'; do for fl in 0 1; do for b in $B/pertensor_gemm_int8_256x256x128_c2x1 $B/pertensor_gemm_fp8_256x256x128_c2x1 $B/pertensor_gemm_int8_256x128x128_c2x1 $B/pertensor_gemm_fp8_256x128x128_c2x1; do \
     \$b \$shp --flush_l2=\$fl --iterations=30 --warmup=5 --verify_samples=256; nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader; done; done; done; \
  python3 - <<'PY'
import torch, statistics
def t(fn, iters=30, flush=None):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts=[]
    for i in range(iters):
        if flush is not None: flush.fill_(i&255)
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e)*1e3)
    return statistics.median(ts)
flush=torch.empty(512<<20,device='cuda',dtype=torch.uint8)
for (m,n,k) in [(8192,8192,8192),(16384,8192,8192),(16384,16384,8192)]:
    a8=torch.randint(-127,128,(m,k),device='cuda',dtype=torch.int8); w8=torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8)
    af=torch.randn(m,k,device='cuda').to(torch.float8_e4m3fn); wf=torch.randn(n,k,device='cuda').to(torch.float8_e4m3fn)
    sa=torch.ones((),device='cuda'); sb=torch.ones((),device='cuda')
    ab=torch.randn(m,k,device='cuda',dtype=torch.bfloat16); wb=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
    fl=2*m*n*k
    for fname,fn in [('int8 _int_mm',lambda: torch._int_mm(a8,w8.t())),('fp8 _scaled_mm per-tensor',lambda: torch._scaled_mm(af,wf.t(),scale_a=sa,scale_b=sb,out_dtype=torch.bfloat16)),('bf16 matmul',lambda: torch.matmul(ab,wb.t()))]:
        for flname,f in [('warm',None),('flush',flush)]:
            us=t(fn,flush=f); print(f'CUBLAS {m}x{n}x{k} {fname:28s} {flname:5s} {us:9.1f} us {fl/us/1e6:7.1f} TFLOPS', flush=True)
    del a8,w8,af,wf,ab,wb; torch.cuda.empty_cache()
PY"
