// Thor microbench: per-SM issue rate of the promotion instructions vs. legacy register-accumulator mma.sync.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include "bench_util.cuh"
constexpr int ITERS = 2048, CH = 32;
#define CHAIN_KERNEL(NAME, DECL, INIT, ASM)                                        \
  __global__ void NAME(float a, float* out) {                                      \
    DECL; _Pragma("unroll") for (int j = 0; j < CH; ++j) { INIT; }                 \
    for (int i = 0; i < ITERS; ++i) { _Pragma("unroll") for (int j = 0; j < CH; ++j) { ASM; } } \
    float s = 0; _Pragma("unroll") for (int j = 0; j < CH; ++j) s += __int_as_float(*(int*)&v[j]); \
    if (s == 123.456f) out[0] = s; }
CHAIN_KERNEL(k_ffma, float v[CH], v[j] = threadIdx.x + j, asm volatile("fma.rn.f32 %0, %1, %0, %0;" : "+f"(v[j]) : "f"(a)))
CHAIN_KERNEL(k_fmul, float v[CH], v[j] = threadIdx.x + j, asm volatile("mul.rn.f32 %0, %1, %0;" : "+f"(v[j]) : "f"(a)))
CHAIN_KERNEL(k_fadd, float v[CH], v[j] = threadIdx.x + j, asm volatile("add.rn.f32 %0, %1, %0;" : "+f"(v[j]) : "f"(a)))
CHAIN_KERNEL(k_i2f,  int v[CH], v[j] = threadIdx.x + j, { float f; asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"(v[j])); v[j] = __float_as_int(f); })
CHAIN_KERNEL(k_f2i,  int v[CH], v[j] = threadIdx.x + j, { float f = __int_as_float(v[j]); asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(v[j]) : "f"(f)); })
CHAIN_KERNEL(k_iadd, int v[CH], v[j] = threadIdx.x + j, asm volatile("add.s32 %0, %0, %1;" : "+r"(v[j]) : "r"(__float_as_int(a))))
CHAIN_KERNEL(k_ffma2, float v[CH], v[j] = threadIdx.x + j, if (j % 2 == 0) asm volatile("fma.rn.f32x2 %0, %1, %0, %0;" : "+l"(*(unsigned long long*)&v[j]) : "l"(*(unsigned long long*)&a)))

__global__ void k_mma_s8(int* out) {
  uint32_t a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, b0 = a0 + 5, b1 = a0 + 7;
  int c[8][4] = {};
  for (int i = 0; i < ITERS; ++i) {
    _Pragma("unroll") for (int j = 0; j < 8; ++j)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  int s = 0; for (int j = 0; j < 8; ++j) for (int k = 0; k < 4; ++k) s += c[j][k];
  if (s == 12345) out[0] = s;
}
__global__ void k_mma_bf16(float* out) {
  uint32_t a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, b0 = a0 + 5, b1 = a0 + 7;
  float c[8][4] = {};
  for (int i = 0; i < ITERS; ++i) {
    _Pragma("unroll") for (int j = 0; j < 8; ++j)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  float s = 0; for (int j = 0; j < 8; ++j) for (int k = 0; k < 4; ++k) s += c[j][k];
  if (s == 12345.f) out[0] = s;
}
int main() {
  cudaDeviceProp p; CUDA_OK(cudaGetDeviceProperties(&p, 0));
  int sms = p.multiProcessorCount, blocks = sms * 4, threads = 256;
  float* fo; int* io; CUDA_OK(cudaMalloc(&fo, 64)); CUDA_OK(cudaMalloc(&io, 64));
  auto run = [&](const char* name, auto fn, double ops_per_thread, bool per_sm) {
    auto t = bench::time_kernel(fn, 5, 20, nullptr);
    int clk = bench::gpu_clock_mhz();
    double ops = ops_per_thread * blocks * threads;
    if (per_sm) printf("%-10s %8.1f us  clk %4d MHz  -> %6.1f ops/clk/SM  (%5.2f per SMSP)\n", name, t.median_us, clk, ops / (t.median_us * 1e-6) / (clk * 1e6) / sms, ops / (t.median_us * 1e-6) / (clk * 1e6) / sms / 4);
    else printf("%-10s %8.1f us  clk %4d MHz  -> %6.1f TOPS  (%6.0f MAC/clk/SM)\n", name, t.median_us, clk, 2 * ops / (t.median_us * 1e-6) / 1e12, ops / (t.median_us * 1e-6) / (clk * 1e6) / sms);
  };
  double n = (double)ITERS * CH;
  run("FFMA", [&] { k_ffma<<<blocks, threads>>>(1.0000001f, fo); }, n, true);
  run("FMUL", [&] { k_fmul<<<blocks, threads>>>(1.0000001f, fo); }, n, true);
  run("FADD", [&] { k_fadd<<<blocks, threads>>>(1.0000001f, fo); }, n, true);
  run("I2F", [&] { k_i2f<<<blocks, threads>>>(1.0f, fo); }, n, true);
  run("F2I", [&] { k_f2i<<<blocks, threads>>>(1.0f, fo); }, n, true);
  run("IADD", [&] { k_iadd<<<blocks, threads>>>(1.0f, fo); }, n, true);
  run("FFMA2(x2)", [&] { k_ffma2<<<blocks, threads>>>(1.0000001f, fo); }, n, true);   // counts 2 flops per f32x2 -> 128/clk if real, 64 if split
  double macs_s8 = (double)ITERS * 8 * (16 * 8 * 32) / 32, macs_bf16 = (double)ITERS * 8 * (16 * 8 * 16) / 32;   // per thread
  run("mma.sync s8", [&] { k_mma_s8<<<blocks, threads>>>(io); }, macs_s8, false);
  run("mma.sync bf16", [&] { k_mma_bf16<<<blocks, threads>>>(fo); }, macs_bf16, false);
  return 0;
}
