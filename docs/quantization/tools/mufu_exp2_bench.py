import torch, triton, triton.language as tl, subprocess
@triton.jit
def k_exp2(x_ptr, out_ptr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)          # BLOCK/threads independent chains per thread (ILP)
    for _ in range(ITERS):
        x = tl.math.exp2(x * 0.001 - 0.5)   # 1 MUFU.EX2 + 1 FFMA per element per iteration
    tl.store(out_ptr + offs, x)
@triton.jit
def k_fma(x_ptr, out_ptr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    for _ in range(ITERS):
        x = x * 0.999 + 0.001               # 1 FFMA per element per iteration
    tl.store(out_ptr + offs, x)
dev = torch.cuda.current_device(); props = torch.cuda.get_device_properties(dev); sms = props.multi_processor_count
def bench(kernel, iters, block, warps, per_elem_ops):
    N = sms * 128 * block
    x = torch.rand(N, device="cuda"); out = torch.empty_like(x)
    grid = (N // block,)
    h = kernel[grid](x, out, ITERS=iters, BLOCK=block, num_warps=warps); torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(5): kernel[grid](x, out, ITERS=iters, BLOCK=block, num_warps=warps)
    clk = int(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
    e1.record(); torch.cuda.synchronize(); ms = e0.elapsed_time(e1) / 5
    rate = N * iters * per_elem_ops / (ms * 1e-3)
    return rate / (sms * clk * 1e6), clk, h
print(f"{props.name} SMs={sms} cc={props.major}.{props.minor}")
for block, warps in [(8192, 8), (16384, 8), (16384, 16), (32768, 16)]:
    r, clk, h = bench(k_exp2, 128, block, warps, 1)
    rf, clkf, _ = bench(k_fma, 512, block, warps, 1)
    print(f"BLOCK={block} warps={warps} ({block//(32*warps)} elems/thread): exp2 {r:.1f}/clk/SM @ {clk} MHz | fma {rf:.1f}/clk/SM (expect 128) | exp2 scaled by fma efficiency {r*128/rf:.1f}")
ptx = h.asm["ptx"]; print("ptx: ex2.approx count", ptx.count("ex2.approx"), "| fma count", ptx.count("fma.rn.f32"), "| other math:", [t for t in ("mul.f32","add.f32","div","rcp") if t in ptx])
