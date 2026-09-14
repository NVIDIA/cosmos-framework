// Shared benchmark utilities for the Thor (sm_110a) GEMM benchmarks: timing with L2 flush, GPU clock sampling,
// random operand init, and a naive device reference GEMM for correctness checks.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_OK(call)                                                                                     \
  do {                                                                                                    \
    cudaError_t _e = (call);                                                                              \
    if (_e != cudaSuccess) {                                                                              \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(_e)); \
      exit(1);                                                                                            \
    }                                                                                                     \
  } while (0)

namespace bench {

// GPU clock in MHz from devfreq (Tegra); -1 if unavailable.
inline int gpu_clock_mhz() {
  std::ifstream f("/sys/class/devfreq/gpu-gpc-0/cur_freq");
  long long hz = -1;
  if (f >> hz) return static_cast<int>(hz / 1000000);
  return -1;
}

// GPU temperature in milli-C from the first thermal zone whose type contains "gpu"; -1 if unavailable.
inline int gpu_temp_mc() {
  for (int i = 0; i < 16; ++i) {
    std::ifstream t("/sys/class/thermal/thermal_zone" + std::to_string(i) + "/type");
    std::string ty;
    if (!(t >> ty)) break;
    if (ty.find("gpu") != std::string::npos || ty.find("GPU") != std::string::npos) {
      std::ifstream v("/sys/class/thermal/thermal_zone" + std::to_string(i) + "/temp");
      int mc = -1;
      v >> mc;
      return mc;
    }
  }
  return -1;
}

struct L2Flush {
  void* buf = nullptr;
  size_t bytes;
  explicit L2Flush(size_t b = size_t(256) << 20) : bytes(b) { CUDA_OK(cudaMalloc(&buf, bytes)); }
  L2Flush(const L2Flush&) = delete;
  L2Flush& operator=(const L2Flush&) = delete;
  ~L2Flush() { cudaFree(buf); }
  void operator()(cudaStream_t s = 0) const { CUDA_OK(cudaMemsetAsync(buf, 0, bytes, s)); }
};

struct TimingResult {
  double median_us = 0, mean_us = 0, min_us = 0;
  int clock_mhz_before = -1, clock_mhz_after = -1;
  std::vector<double> all_us;  // per-iteration times in launch order
};

// Times fn() `iters` times, flushing L2 before each iteration if flush != nullptr (each iteration timed by its
// own event pair so the flush is excluded). Returns median / mean / min in microseconds.
// Warmup runs at least `warmup` iterations AND at least `min_warmup_ms` of wall time: the Thor GPU idles at 315 MHz and
// the devfreq governor needs a few hundred ms of load to reach the 1386 MHz cap (120 W mode).
template <class Fn>
TimingResult time_kernel(Fn&& fn, int warmup, int iters, const L2Flush* flush, cudaStream_t stream = 0,
                         double min_warmup_ms = 300.0) {
  TimingResult r;
  auto t0 = std::chrono::steady_clock::now();
  int done = 0;
  while (true) {
    for (int i = 0; i < 10; ++i, ++done) fn();
    CUDA_OK(cudaStreamSynchronize(stream));
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (done >= warmup && ms >= min_warmup_ms) break;
  }
  r.clock_mhz_before = gpu_clock_mhz();
  std::vector<cudaEvent_t> ev(2 * iters);
  for (auto& e : ev) CUDA_OK(cudaEventCreate(&e));
  for (int i = 0; i < iters; ++i) {
    if (flush) (*flush)(stream);
    CUDA_OK(cudaEventRecord(ev[2 * i], stream));
    fn();
    CUDA_OK(cudaEventRecord(ev[2 * i + 1], stream));
  }
  CUDA_OK(cudaStreamSynchronize(stream));
  r.clock_mhz_after = gpu_clock_mhz();
  std::vector<double> t(iters);
  for (int i = 0; i < iters; ++i) {
    float ms = 0;
    CUDA_OK(cudaEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1]));
    t[i] = ms * 1e3;
  }
  for (auto& e : ev) CUDA_OK(cudaEventDestroy(e));
  r.all_us = t;
  std::sort(t.begin(), t.end());
  r.min_us = t.front();
  r.median_us = (iters % 2) ? t[iters / 2] : 0.5 * (t[iters / 2 - 1] + t[iters / 2]);
  double s = 0;
  for (double v : t) s += v;
  r.mean_us = s / iters;
  return r;
}

inline double tflops(double flop_count, double us) { return flop_count / (us * 1e-6) / 1e12; }

// ---------------------------------------------------------------- random init kernels
__global__ void fill_int8_kernel(int8_t* p, size_t n, uint32_t seed, int lo, int hi) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t x = (uint32_t)i * 2654435761u ^ seed;
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  p[i] = (int8_t)(lo + (int)(x % (uint32_t)(hi - lo + 1)));
}
__global__ void fill_e4m3_kernel(__nv_fp8_e4m3* p, size_t n, uint32_t seed, float lo, float hi) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t x = (uint32_t)i * 2654435761u ^ seed;
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  float v = lo + (hi - lo) * (float)(x & 0xffffff) / 16777216.f;
  p[i] = __nv_fp8_e4m3(v);
}
__global__ void fill_bf16_kernel(__nv_bfloat16* p, size_t n, uint32_t seed, float lo, float hi) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t x = (uint32_t)i * 2654435761u ^ seed;
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  float v = lo + (hi - lo) * (float)(x & 0xffffff) / 16777216.f;
  p[i] = __float2bfloat16(v);
}
// "Realistic" operands: x ~ N(0,1) (Box-Muller on two hash uniforms), then per-tensor quantized with absmax taken as 5 sigma:
// int8 q = rint(x * 127/5) clamped to [-127,127]; e4m3 v = x * 448/5 (e4m3 max = 448). Mimics a per-tensor-quantized Gaussian tensor.
__device__ inline float hash_normal(size_t i, uint32_t seed) {
  uint32_t x = (uint32_t)i * 2654435761u ^ seed;
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  uint32_t y = x * 0x9E3779B1u ^ 0x85ebca6bu;
  y ^= y >> 13; y *= 0xc2b2ae35u; y ^= y >> 16;
  float u1 = ((x >> 8) + 1.f) / 16777217.f, u2 = (y >> 8) / 16777216.f;
  return sqrtf(-2.f * logf(u1)) * cosf(6.283185307f * u2);
}
__global__ void fill_int8_normal_kernel(int8_t* p, size_t n, uint32_t seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  float q = rintf(hash_normal(i, seed) * (127.f / 5.f));
  p[i] = (int8_t)fminf(127.f, fmaxf(-127.f, q));
}
__global__ void fill_e4m3_normal_kernel(__nv_fp8_e4m3* p, size_t n, uint32_t seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = hash_normal(i, seed) * (448.f / 5.f);
  p[i] = __nv_fp8_e4m3(fminf(448.f, fmaxf(-448.f, v)));
}
inline void fill_int8_normal(int8_t* p, size_t n, uint32_t seed) {
  fill_int8_normal_kernel<<<(unsigned)((n + 255) / 256), 256>>>(p, n, seed);
  CUDA_OK(cudaGetLastError());
}
inline void fill_e4m3_normal(__nv_fp8_e4m3* p, size_t n, uint32_t seed) {
  fill_e4m3_normal_kernel<<<(unsigned)((n + 255) / 256), 256>>>(p, n, seed);
  CUDA_OK(cudaGetLastError());
}
inline void fill_int8(int8_t* p, size_t n, uint32_t seed, int lo = -127, int hi = 127) {
  fill_int8_kernel<<<(unsigned)((n + 255) / 256), 256>>>(p, n, seed, lo, hi);
  CUDA_OK(cudaGetLastError());
}
inline void fill_e4m3(__nv_fp8_e4m3* p, size_t n, uint32_t seed, float lo = -2.f, float hi = 2.f) {
  fill_e4m3_kernel<<<(unsigned)((n + 255) / 256), 256>>>(p, n, seed, lo, hi);
  CUDA_OK(cudaGetLastError());
}
inline void fill_bf16(__nv_bfloat16* p, size_t n, uint32_t seed, float lo = -2.f, float hi = 2.f) {
  fill_bf16_kernel<<<(unsigned)((n + 255) / 256), 256>>>(p, n, seed, lo, hi);
  CUDA_OK(cudaGetLastError());
}

// ---------------------------------------------------------------- naive reference: D[m,n] = alpha * sum_k A[m,k] * W[n,k]
// A[M,K] row-major, W[N,K] row-major (both K-contiguous). Output fp32 (int32 math exact for int8; fp32 accumulate for fp8).
template <class TIn, class TAcc>
__global__ void ref_gemm_kernel(const TIn* A, const TIn* W, float* D, int M, int N, int K, float alpha) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  TAcc acc = TAcc(0);
  const TIn* a = A + (size_t)m * K;
  const TIn* w = W + (size_t)n * K;
  for (int k = 0; k < K; ++k) acc += TAcc(float(a[k])) * TAcc(float(w[k]));
  D[(size_t)m * N + n] = alpha * float(acc);
}
inline void ref_gemm_int8(const int8_t* A, const int8_t* W, float* D, int M, int N, int K, float alpha) {
  dim3 b(32, 8), g((N + 31) / 32, (M + 7) / 8);
  ref_gemm_kernel<int8_t, int32_t><<<g, b>>>(A, W, D, M, N, K, alpha);
  CUDA_OK(cudaGetLastError());
}
inline void ref_gemm_e4m3(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* W, float* D, int M, int N, int K, float alpha) {
  dim3 b(32, 8), g((N + 31) / 32, (M + 7) / 8);
  ref_gemm_kernel<__nv_fp8_e4m3, float><<<g, b>>>(A, W, D, M, N, K, alpha);
  CUDA_OK(cudaGetLastError());
}
inline void ref_gemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* W, float* D, int M, int N, int K, float alpha) {
  dim3 b(32, 8), g((N + 31) / 32, (M + 7) / 8);
  ref_gemm_kernel<__nv_bfloat16, float><<<g, b>>>(A, W, D, M, N, K, alpha);
  CUDA_OK(cudaGetLastError());
}

// rel-L2 and max-abs error of bf16 output vs fp32 reference (computed on host).
struct ErrStats { double rel_l2 = 0, max_abs = 0, ref_max_abs = 0; };
inline ErrStats compare_bf16_vs_f32(const __nv_bfloat16* d_out, const float* d_ref, size_t n) {
  std::vector<__nv_bfloat16> o(n);
  std::vector<float> r(n);
  CUDA_OK(cudaMemcpy(o.data(), d_out, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(r.data(), d_ref, n * sizeof(float), cudaMemcpyDeviceToHost));
  double num = 0, den = 0;
  ErrStats e;
  for (size_t i = 0; i < n; ++i) {
    double v = (double)__bfloat162float(o[i]), ref = (double)r[i], d = v - ref;
    num += d * d; den += ref * ref;
    e.max_abs = std::max(e.max_abs, std::abs(d));
    e.ref_max_abs = std::max(e.ref_max_abs, std::abs(ref));
  }
  e.rel_l2 = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  return e;
}
inline ErrStats compare_i32_vs_f32(const int32_t* d_out, const float* d_ref, size_t n) {
  std::vector<int32_t> o(n);
  std::vector<float> r(n);
  CUDA_OK(cudaMemcpy(o.data(), d_out, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(r.data(), d_ref, n * sizeof(float), cudaMemcpyDeviceToHost));
  double num = 0, den = 0;
  ErrStats e;
  for (size_t i = 0; i < n; ++i) {
    double v = (double)o[i], ref = (double)r[i], d = v - ref;
    num += d * d; den += ref * ref;
    e.max_abs = std::max(e.max_abs, std::abs(d));
    e.ref_max_abs = std::max(e.ref_max_abs, std::abs(ref));
  }
  e.rel_l2 = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  return e;
}

// Minimal CLI parsing: --key=value
struct Args {
  std::vector<std::string> kv;
  Args(int argc, char** argv) { for (int i = 1; i < argc; ++i) kv.emplace_back(argv[i]); }
  std::string get(const std::string& key, const std::string& def) const {
    std::string pre = "--" + key + "=";
    for (auto& s : kv) if (s.rfind(pre, 0) == 0) return s.substr(pre.size());
    return def;
  }
  int geti(const std::string& key, int def) const { return std::stoi(get(key, std::to_string(def))); }
  bool has(const std::string& key) const { for (auto& s : kv) if (s == "--" + key) return true; return get(key, "") != ""; }
};

}  // namespace bench
