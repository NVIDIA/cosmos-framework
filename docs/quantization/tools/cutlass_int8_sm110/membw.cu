// Memory-system microbenchmark for Thor: aggregate L2 read bandwidth (all SMs re-reading an L2-resident buffer) and DRAM
// read bandwidth (streaming a buffer much larger than L2), plus the peak tensor-op check via a smem-resident-free estimate.
//   nvcc -O3 -arch=sm_110a -o membw membw.cu && ./membw
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include "bench_util.cuh"

// Each block streams `bytes_per_block` bytes starting at a block-dependent offset within the buffer (wrapping), 16B per thread per iter.
__global__ void read_kernel(const int4* __restrict__ buf, size_t n_int4, size_t iters_per_thread, int4* sink) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * (size_t)blockDim.x;
  int4 acc = make_int4(0, 0, 0, 0);
  size_t idx = tid;
  for (size_t i = 0; i < iters_per_thread; ++i) {
    int4 v = __ldcg(buf + idx);
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    idx += stride;
    if (idx >= n_int4) idx -= n_int4;
  }
  if (acc.x == 0x7fffffff && acc.y == 0x7ffffffe) sink[0] = acc;  // never true; defeats DCE
}

static double run_read(size_t bytes, size_t total_read_bytes, int blocks, int threads, bench::L2Flush* flush) {
  int4* buf; int4* sink;
  CUDA_OK(cudaMalloc(&buf, bytes));
  CUDA_OK(cudaMalloc(&sink, 16));
  CUDA_OK(cudaMemset(buf, 1, bytes));
  size_t n_int4 = bytes / 16;
  size_t iters = total_read_bytes / 16 / ((size_t)blocks * threads);
  auto fn = [&] { read_kernel<<<blocks, threads>>>(buf, n_int4, iters, sink); };
  // warm the buffer into L2 (if it fits) by running once; flush only for the DRAM case.
  auto t = bench::time_kernel(fn, 3, 10, flush);
  double gbps = (double)iters * 16 * blocks * threads / (t.median_us * 1e-6) / 1e9;
  CUDA_OK(cudaFree(buf)); CUDA_OK(cudaFree(sink));
  return gbps;
}

int main() {
  cudaDeviceProp p; CUDA_OK(cudaGetDeviceProperties(&p, 0));
  printf("%s SMs=%d L2=%d MB clk=%d MHz\n", p.name, p.multiProcessorCount, p.l2CacheSize >> 20, bench::gpu_clock_mhz());
  bench::L2Flush flush;
  for (int blocks_per_sm : {4, 8, 16}) {
    int blocks = p.multiProcessorCount * blocks_per_sm;
    double l2_8 = run_read(8u << 20, size_t(4) << 30, blocks, 256, nullptr);
    double l2_16 = run_read(16u << 20, size_t(4) << 30, blocks, 256, nullptr);
    double dram = run_read(size_t(1) << 30, size_t(1) << 30, blocks, 256, &flush);
    printf("blocks/SM=%2d  L2 read (8 MB buf) %.0f GB/s | L2 read (16 MB buf) %.0f GB/s | DRAM read (1 GB) %.0f GB/s  [clk %d MHz]\n",
           blocks_per_sm, l2_8, l2_16, dram, bench::gpu_clock_mhz());
  }
  // D2D copy bandwidth (read+write)
  {
    void *a, *b; size_t bytes = size_t(1) << 30;
    CUDA_OK(cudaMalloc(&a, bytes)); CUDA_OK(cudaMalloc(&b, bytes));
    auto t = bench::time_kernel([&] { CUDA_OK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice, 0)); }, 2, 5, nullptr);
    printf("cudaMemcpy D2D 1 GB: %.0f GB/s (counting read+write)\n", 2.0 * bytes / (t.median_us * 1e-6) / 1e9);
    CUDA_OK(cudaFree(a)); CUDA_OK(cudaFree(b));
  }
  return 0;
}
