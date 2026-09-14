"""Pooled weight SNR / GEMM proxy for INT8 group sizes 64/128/256 on the 252 Cosmos3-Nano gen-tower linears (same math as weight_scale_study.py)."""
import json, os, re, time, torch
from safetensors import safe_open
SNAP='/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/hf_cache/hub/models--nvidia--Cosmos3-Nano/snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa'
idx=json.load(open(f'{SNAP}/model.safetensors.index.json'))['weight_map']
pat=re.compile(r'layers\.(\d+)\.(self_attn\.(add_q_proj|add_k_proj|add_v_proj|to_add_out)|mlp_moe_gen\.(gate_proj|up_proj|down_proj))\.weight$')
names=sorted(k for k in idx if pat.search(k)); assert len(names)==252, len(names)
dev='cuda'; Q=127.0
def q_int8(w,scale): return torch.round(w/scale).clamp_(-Q,Q)*scale
def percol(w,g):
    N,K=w.shape; w3=w.view(N,K//g,g); s=(w3.abs().amax(-1,keepdim=True)/Q).clamp_min(1e-30); return q_int8(w3,s).view(N,K)
def block(w,bn,bk):
    N,K=w.shape; w4=w.view(N//bn,bn,K//bk,bk).permute(0,2,1,3); s=(w4.abs().amax((-1,-2),keepdim=True)/Q).clamp_min(1e-30); return q_int8(w4,s).permute(0,2,1,3).reshape(N,K)
def sep_h(w,g):
    N,K=w.shape; w3=w.view(N,K//g,g); A=w3.abs().amax(-1).clamp_min(1e-30); la=A.log2(); lsw=la.mean(1,keepdim=True); c=(la-lsw).mean(0,keepdim=True).exp2()
    sw=(A/c).amax(1,keepdim=True); s=(sw*c/Q).unsqueeze(-1); return q_int8(w3,s).view(N,K)
def perchan(w): s=(w.abs().amax(1,keepdim=True)/Q); return q_int8(w,s)
schemes={'percol_g64':lambda w:percol(w,64),'percol_g128':lambda w:percol(w,128),'percol_g256':lambda w:percol(w,256),
         'block128x128':lambda w:block(w,128,128),'block256x256':lambda w:block(w,256,256),
         'sep_h_g128':lambda w:sep_h(w,128),'sep_h_g256':lambda w:sep_h(w,256),'perchan':perchan}
err={k:0.0 for k in schemes}; gerr={k:0.0 for k in schemes}; wtot=0.0; gtot=0.0; Xs={}
t0=time.time(); shards=sorted(set(idx[n] for n in names))
for sh in shards:
    with safe_open(f'{SNAP}/{sh}','pt',device='cpu') as f:
        for n in names:
            if idx[n]!=sh: continue
            w=f.get_tensor(n).to(dev).float(); N,K=w.shape
            if K not in Xs: Xs[K]=torch.randn(2048,K,device=dev,generator=torch.Generator(device=dev).manual_seed(K)).to(torch.bfloat16).float()
            X=Xs[K]; torch.backends.cuda.matmul.allow_tf32=False; Yr=X@w.t(); wtot+=(w*w).sum().item(); gtot+=(Yr*Yr).sum().item()
            for k,fn in schemes.items():
                d=fn(w)-w; err[k]+=(d*d).sum().item(); e=X@d.t(); gerr[k]+=(e*e).sum().item()
    print(f'[{time.time()-t0:5.0f}s] done {sh}', flush=True)
import math
print('\n| scheme | weight SNR dB (pooled) | GEMM-proxy rel-L2 |'); print('|---|---|---|')
for k in schemes: print(f'| {k} | {10*math.log10(wtot/err[k]):.2f} | {math.sqrt(gerr[k]/gtot):.4f} |')
