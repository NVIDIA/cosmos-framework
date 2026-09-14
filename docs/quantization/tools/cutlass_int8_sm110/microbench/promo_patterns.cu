// Issue cost per element of the g128 promotion patterns (register-only, no TMEM): clk per element per SMSP.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include "bench_util.cuh"
constexpr int ITERS = 2048, CH = 32;
#define PAT_KERNEL(NAME, BODY)                                                                     \
  __global__ void NAME(float sa, float sw, float c, int M, float* out) {                           \
    float acc[CH]; int x[CH];                                                                      \
    _Pragma("unroll") for (int j = 0; j < CH; ++j) { acc[j] = 0.f; x[j] = threadIdx.x * 7 + j; }   \
    for (int i = 0; i < ITERS; ++i) { _Pragma("unroll") for (int j = 0; j < CH; ++j) { BODY; } }   \
    float s = 0; _Pragma("unroll") for (int j = 0; j < CH; ++j) s += acc[j];                        \
    if (s == 123.456f) out[0] = s; }
// P1: per-col today: I2F, FMUL, FFMA
PAT_KERNEL(p1_percol_now, { float f; asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"(x[j])); float p; asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(p) : "f"(f), "f"(sa)); asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[j]) : "f"(p), "f"(sw)); x[j] = __float_as_int(acc[j]); })
// P2: W-block today: I2F, FFMA
PAT_KERNEL(p2_wblock_now, { float f; asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"(x[j])); asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[j]) : "f"(f), "f"(sa)); x[j] = __float_as_int(acc[j]); })
// P3: TMEM pre-biased accumulator, per-col: FFMA(fb, sa, c) then FFMA(t, sw, acc) -- 2 instructions
PAT_KERNEL(p3_percol_bias_tmem, { float fb = __int_as_float(x[j]); float t; asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(t) : "f"(fb), "f"(sa), "f"(c)); asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[j]) : "f"(t), "f"(sw)); x[j] = __float_as_int(acc[j]); })
// P4: bias added in registers (IADD) then the two FFMAs -- no TMEM change needed
PAT_KERNEL(p4_percol_bias_reg, { int xb; asm volatile("add.s32 %0, %1, %2;" : "=r"(xb) : "r"(x[j]), "r"(M)); float fb = __int_as_float(xb); float t; asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(t) : "f"(fb), "f"(sa), "f"(c)); asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[j]) : "f"(t), "f"(sw)); x[j] = __float_as_int(acc[j]); })
// P5: TMEM pre-biased, W-block: FFMA(fb, s, c) then FADD
PAT_KERNEL(p5_wblock_bias_tmem, { float fb = __int_as_float(x[j]); float t; asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(t) : "f"(fb), "f"(sa), "f"(c)); asm volatile("add.rn.f32 %0, %1, %0;" : "+f"(acc[j]) : "f"(t)); x[j] = __float_as_int(acc[j]); })
int main() {
  cudaDeviceProp p; CUDA_OK(cudaGetDeviceProperties(&p, 0));
  int sms = p.multiProcessorCount, blocks = sms * 4, threads = 256;
  float* fo; CUDA_OK(cudaMalloc(&fo, 64));
  auto run = [&](const char* name, auto kern) {
    auto fn = [&] { kern<<<blocks, threads>>>(1.0001f, 0.9999f, -12582912.f * 1.0001f, 0x4B400000, fo); };
    auto t = bench::time_kernel(fn, 5, 20, nullptr);
    int clk = bench::gpu_clock_mhz();
    double elems = (double)ITERS * CH * blocks * threads;                 // element-promotions
    double clk_per_elem_per_smsp = (t.median_us * 1e-6 * clk * 1e6) * sms * 4 * 32 / elems;   // 32 lanes per SMSP
    printf("%-24s %8.1f us  clk %4d  -> %.2f issue-clk per element (per lane)\n", name, t.median_us, clk, clk_per_elem_per_smsp);
  };
  run("P1 per-col now", p1_percol_now); run("P2 W-block now", p2_wblock_now);
  run("P3 per-col bias(TMEM)", p3_percol_bias_tmem); run("P4 per-col bias(IADD)", p4_percol_bias_reg); run("P5 W-block bias(TMEM)", p5_wblock_bias_tmem);
  // exactness check of the bias trick for the INT8 g128 range
  long bad = 0; for (long x = -2064512; x <= 2064512; x += 1) { int b = 0x4B400000 + (int)x; float f; memcpy(&f, &b, 4); if ((double)f - 12582912.0 != (double)x) ++bad; }
  printf("bias-trick exactness over |x| <= 128*127*127: %ld mismatches\n", bad);
  return 0;
}
