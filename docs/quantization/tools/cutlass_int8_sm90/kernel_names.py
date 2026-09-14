import torch
from torch.profiler import profile, ProfilerActivity
dev="cuda"; M=N=K=4096
A=torch.randn(M,K,device=dev,dtype=torch.bfloat16); W=torch.randn(N,K,device=dev,dtype=torch.bfloat16)
Aq=torch.randint(-127,128,(M,K),device=dev,dtype=torch.int8); Wq=torch.randint(-127,128,(N,K),device=dev,dtype=torch.int8)
Af=A.to(torch.float8_e4m3fn); Wf=W.to(torch.float8_e4m3fn); one=torch.ones((),device=dev)
fns={"bf16 cuBLAS": lambda: A@W.t(),
     "INT8 _int_mm": lambda: torch._int_mm(Aq,Wq.t()),
     "FP8 _scaled_mm per-tensor": lambda: torch._scaled_mm(Af,Wf.t(),scale_a=one,scale_b=one,out_dtype=torch.bfloat16,use_fast_accum=True),
     "FP8 _scaled_mm rowwise": lambda: torch._scaled_mm(Af,Wf.t(),scale_a=torch.ones(M,1,device=dev),scale_b=torch.ones(1,N,device=dev),out_dtype=torch.bfloat16,use_fast_accum=True)}
for name,fn in fns.items():
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3): fn()
        torch.cuda.synchronize()
    ev=[(e.key,e.device_time_total/3) for e in p.key_averages() if e.device_time_total>0]
    ev.sort(key=lambda x:-x[1])
    print(f"\n{name}:")
    for k,t in ev[:3]: print(f"   {t:7.0f} us  {k[:150]}")
