// MUFU.EX2 (exp2f) and FFMA throughput per SM per clock.  Build & run (e.g. on Jetson Thor):
//   nvcc -O3 -use_fast_math -arch=native -o mufu_exp2_bench mufu_exp2_bench.cu && ./mufu_exp2_bench
// exp2f with -use_fast_math compiles to ex2.approx.f32 (one MUFU.EX2). 32 independent chains per thread give ILP.
#include <cstdio>
#include <cuda_runtime.h>
#define CHAINS 32
template <bool EXP>
__global__ void bench(float* out, int iters, float a, float b) {
    float x[CHAINS];
    #pragma unroll
    for (int c = 0; c < CHAINS; ++c) x[c] = 0.001f * (threadIdx.x + c);
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int c = 0; c < CHAINS; ++c) x[c] = EXP ? exp2f(x[c] * a + b) : (x[c] * a + b);  // 1 MUFU (+1 FFMA) or 1 FFMA
    }
    float s = 0.f;
    #pragma unroll
    for (int c = 0; c < CHAINS; ++c) s += x[c];
    if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;  // keep the work alive
}
template <bool EXP>
double run(int sms, int blocks_per_sm, int threads, int iters) {
    float* out; cudaMalloc(&out, sizeof(float) * sms * blocks_per_sm * threads);
    bench<EXP><<<sms * blocks_per_sm, threads>>>(out, iters, 0.001f, -0.5f); cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int r = 0; r < 5; ++r) bench<EXP><<<sms * blocks_per_sm, threads>>>(out, iters, 0.001f, -0.5f);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
    double ops = (double)sms * blocks_per_sm * threads * iters * CHAINS;
    cudaFree(out);
    return ops / (ms * 1e-3);
}
int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    int clk_khz = 0; cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    double clk = clk_khz * 1e3;  // nominal max SM clock; compare with the actual clock (nvidia-smi / tegrastats) if it throttles
    printf("%s  cc %d.%d  SMs %d  clock %.0f MHz (nominal)\n", p.name, p.major, p.minor, p.multiProcessorCount, clk / 1e6);
    for (int bps : {8, 16, 32}) {
        double ex = run<true>(p.multiProcessorCount, bps, 256, 64);
        double fm = run<false>(p.multiProcessorCount, bps, 256, 256);
        printf("blocks/SM %2d: exp2 %.1f /clk/SM   ffma %.1f /clk/SM (expect 128)   exp2 scaled by ffma efficiency %.1f\n",
               bps, ex / (p.multiProcessorCount * clk), fm / (p.multiProcessorCount * clk), ex / fm * 128.0);
    }
    return 0;
}
