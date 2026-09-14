import sys, os, json, numpy as np
from PIL import Image
ref_dir, run_dir = sys.argv[1], sys.argv[2]
rows=[]
for name in sorted(os.listdir(ref_dir)):
    a=os.path.join(ref_dir,name,'vision.jpg'); b=os.path.join(run_dir,name,'vision.jpg')
    if not (os.path.exists(a) and os.path.exists(b)): continue
    x=np.asarray(Image.open(a).convert('RGB'),dtype=np.float64); y=np.asarray(Image.open(b).convert('RGB'),dtype=np.float64)
    mse=((x-y)**2).mean(); psnr=99.0 if mse==0 else 10*np.log10(255.0**2/mse); rows.append((name,float(psnr)))
m=np.mean([p for _,p in rows]) if rows else float('nan'); kept=int(sum(p>=22 for _,p in rows)); m=float(m)
print(f"{os.path.basename(run_dir):18s} n={len(rows)} mean={m:6.2f} dB  min={min(p for _,p in rows):6.2f}  kept(>=22dB)={kept}/{len(rows)}  | " + " ".join(f"{n}:{p:.1f}" for n,p in rows))
json.dump({'run':run_dir,'rows':rows,'mean':m,'kept':kept}, open(os.path.join(run_dir,'psnr_vs_bf16.json'),'w'))
