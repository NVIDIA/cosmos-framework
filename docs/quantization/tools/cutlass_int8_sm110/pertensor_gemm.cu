// Benchmark / correctness driver for the CUTLASS per-tensor FP8 and INT8 GEMMs on Thor (sm_110a).
// Usage: ./pertensor_gemm --dtype=int8|fp8 --cfg=0 --m=4096 --n=4096 --k=4096 [--iters=50 --warmup=10 --flush=1 --verify=1
//                         --sa=0.0123 --sb=0.0456 --swizzle=auto|<int> --raster=H|M|N --device_scale=0 --nw=1
//                         --scale=tensor|rowcol]   |  --list      (rowcol: D = acc * sa[m] * sb[n], fp32 vectors, cfg 2/3 only)
// --nw=N: allocate N distinct weight copies and rotate through them, one per timed iteration. --flush=0 --nw=8 models the
// production layer loop on Thor (activations hot in the 32 MB L2, every layer's weight cold in DRAM); --flush=1 is all-cold.
// swizzle=auto: 1 (off) when W = N*K bytes <= 16 MiB (fits L2, swizzle measured as a no-op or slightly negative); otherwise the largest
// power of two g <= 16 such that g * TileM * K bytes (the A row-blocks of one raster group) <= 16 MiB, i.e. half of Thor's 32 MB L2,
// so W column-blocks are streamed from DRAM once per group instead of once per M row-block.
// Output (stdout, one tab-separated line): lib=cutlass dtype cfg M N K median_us mean_us min_us tflops clk_before clk_after verify rel_l2 desc
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "bench_util.cuh"
#include "pertensor_gemm.cuh"
#include "configs.h"

#define DECL_I8(I, TM, TN, TK, CM, CN, S, E, D) THOR_DECL(int8, I)
#define DECL_F8(I, TM, TN, TK, CM, CN, S, E, D) THOR_DECL(fp8, I)
THOR_CFG_LIST(DECL_I8)
THOR_CFG_LIST(DECL_F8)
#define DECL_RC_I8(I, TM, TN, TK, CM, CN, S, E, D) THOR_DECL_RC(int8, I)
#define DECL_RC_F8(I, TM, TN, TK, CM, CN, S, E, D) THOR_DECL_RC(fp8, I)
THOR_RC_CFG_LIST(DECL_RC_I8)
THOR_RC_CFG_LIST(DECL_RC_F8)

using Factory = thor::GemmHandle* (*)();
static Factory kInt8[] = {
#define F_I8(I, TM, TN, TK, CM, CN, S, E, D) thor_make_int8_##I,
    THOR_CFG_LIST(F_I8)
};
static Factory kFp8[] = {
#define F_F8(I, TM, TN, TK, CM, CN, S, E, D) thor_make_fp8_##I,
    THOR_CFG_LIST(F_F8)
};
static Factory rc_factory(bool is_int8, int cfg) {
  switch (cfg) {
#define RC_CASE(I, TM, TN, TK, CM, CN, S, E, D) case I: return is_int8 ? thor_make_rc_int8_##I : thor_make_rc_fp8_##I;
    THOR_RC_CFG_LIST(RC_CASE)
    default: return nullptr;
  }
}
// ref[i, j] *= sa[i] * sb[j]
__global__ void scale_rowcol_kernel(float* ref, int rows, int N, const float* sa, const float* sb) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < (size_t)rows * N) ref[i] *= sa[i / N] * sb[i % N];
}
static const char* kDesc[] = {
#define D_(I, TM, TN, TK, CM, CN, S, E, D) D,
    THOR_CFG_LIST(D_)
};

int main(int argc, char** argv) {
  bench::Args args(argc, argv);
  if (args.has("list")) {
    for (int i = 0; i < THOR_NUM_CFGS; ++i) printf("cfg%d: %s\n", i, kDesc[i]);
    return 0;
  }
  std::string dtype = args.get("dtype", "int8");
  int cfg = args.geti("cfg", 0), M = args.geti("m", 4096), N = args.geti("n", 4096), K = args.geti("k", 4096);
  int iters = args.geti("iters", 50), warmup = args.geti("warmup", 10), flush = args.geti("flush", 1);
  int verify = args.geti("verify", 1), device_scale = args.geti("device_scale", 0), nw = std::max(1, args.geti("nw", 1));
  int zeros = args.geti("zeros", 0);  // 1: A and W all zero (power / data-dependence probe; verify is skipped)
  std::string dist = args.get("dist", "uniform");  // uniform (full-range) | normal (per-tensor-quantized Gaussian, absmax = 5 sigma)
  double warmup_ms = std::stod(args.get("warmup_ms", "300"));  // minimum warm-up wall time (0 = start timing from a cold GPU)
  std::string dump = args.get("dump", "");  // write per-iteration microseconds (launch order) to this file
  std::string swz = args.get("swizzle", "auto"), ras = args.get("raster", "H");
  int raster = ras == "M" ? 1 : ras == "N" ? 2 : 0;
  float sa = std::stof(args.get("sa", "0.0123")), sb = std::stof(args.get("sb", "0.0456"));
  std::string scale = args.get("scale", "tensor");  // tensor: D = acc*sa*sb | rowcol: D = acc*sa[m]*sb[n] (fp32 device vectors)
  bool rowcol = scale == "rowcol";
  if (!rowcol && scale != "tensor") { fprintf(stderr, "scale must be tensor|rowcol\n"); return 1; }
  if (cfg < 0 || cfg >= THOR_NUM_CFGS) { fprintf(stderr, "bad cfg\n"); return 1; }
  bool is_int8 = dtype == "int8";
  if (!is_int8 && dtype != "fp8") { fprintf(stderr, "dtype must be int8|fp8\n"); return 1; }

  size_t nA = (size_t)M * K, nB = (size_t)N * K, nD = (size_t)M * N;
  void *A = nullptr, *Ball = nullptr;
  __nv_bfloat16* D = nullptr;
  float* d_scales = nullptr;
  CUDA_OK(cudaMalloc(&A, nA));
  CUDA_OK(cudaMalloc(&Ball, nB * nw));
  void* B = Ball;  // weight copy 0 (used for verification)
  CUDA_OK(cudaMalloc(&D, nD * sizeof(__nv_bfloat16)));
  CUDA_OK(cudaMalloc(&d_scales, 2 * sizeof(float)));
  float h_scales[2] = {sa, sb};
  CUDA_OK(cudaMemcpy(d_scales, h_scales, sizeof(h_scales), cudaMemcpyHostToDevice));
  float *d_sa_vec = nullptr, *d_sb_vec = nullptr;   // row x col scale vectors (sa[m], sb[n])
  if (rowcol) {
    CUDA_OK(cudaMalloc(&d_sa_vec, (size_t)M * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_sb_vec, (size_t)N * sizeof(float)));
    bench::fill_scales(d_sa_vec, M, 0x77u);
    bench::fill_scales(d_sb_vec, N, 0x99u);
  }
  if (zeros) {
    CUDA_OK(cudaMemset(A, 0, nA));
    CUDA_OK(cudaMemset(Ball, 0, nB * nw));
    verify = 0;
  } else if (is_int8 && dist == "normal") {
    bench::fill_int8_normal((int8_t*)A, nA, 0x1234u);
    for (int w = 0; w < nw; ++w) bench::fill_int8_normal((int8_t*)Ball + (size_t)w * nB, nB, 0x5678u + 977u * w);
  } else if (is_int8) {
    bench::fill_int8((int8_t*)A, nA, 0x1234u);
    for (int w = 0; w < nw; ++w) bench::fill_int8((int8_t*)Ball + (size_t)w * nB, nB, 0x5678u + 977u * w);
  } else if (dist == "normal") {
    bench::fill_e4m3_normal((__nv_fp8_e4m3*)A, nA, 0x1234u);
    for (int w = 0; w < nw; ++w) bench::fill_e4m3_normal((__nv_fp8_e4m3*)Ball + (size_t)w * nB, nB, 0x5678u + 977u * w);
  } else {
    bench::fill_e4m3((__nv_fp8_e4m3*)A, nA, 0x1234u);
    for (int w = 0; w < nw; ++w) bench::fill_e4m3((__nv_fp8_e4m3*)Ball + (size_t)w * nB, nB, 0x5678u + 977u * w);
  }
  CUDA_OK(cudaMemset(D, 0, nD * sizeof(__nv_bfloat16)));
  CUDA_OK(cudaDeviceSynchronize());

  // one handle per weight copy (same kernel, different B pointer); h = handle 0
  std::vector<std::unique_ptr<thor::GemmHandle>> hs;
  Factory fac = rowcol ? rc_factory(is_int8, cfg) : (is_int8 ? kInt8 : kFp8)[cfg];
  if (!fac) { fprintf(stderr, "cfg%d is not instantiated with --scale=rowcol (see THOR_RC_CFG_LIST)\n", cfg); return 1; }
  for (int w = 0; w < nw; ++w) hs.emplace_back(fac());
  thor::GemmHandle* h = hs[0].get();
  int swizzle = 0;
  if (swz == "auto") {
    // Swizzle only pays when the streamed operand (W with the scheduler's default N-fastest order) does not fit in L2; with W <= 16 MiB
    // it is a no-op or 1-10 % slower (measured on 4096x4096 / 1024x4096). Then group as many M row-blocks as keep A's group <= 16 MiB.
    swizzle = 1;
    if ((double)N * K > 16.0 * 1024 * 1024) {
      double budget = 16.0 * 1024 * 1024 / ((double)h->tile_m() * K);
      while (swizzle * 2 <= budget && swizzle < 16) swizzle *= 2;
    }
  } else {
    swizzle = std::stoi(swz);
  }
  try {
    for (int w = 0; w < nw; ++w)
      hs[w]->init(A, (char*)Ball + (size_t)w * nB, D, M, N, K, sa, sb, rowcol ? d_sa_vec : device_scale ? d_scales : nullptr,
                  rowcol ? d_sb_vec : device_scale ? d_scales + 1 : nullptr, swizzle, raster, 0);
    h->run(0);
    CUDA_OK(cudaDeviceSynchronize());
  } catch (std::exception const& e) {
    printf("lib=cutlass\tdtype=%s\tcfg=%d\tM=%d\tN=%d\tK=%d\tERROR=%s\tdesc=%s\n", dtype.c_str(), cfg, M, N, K, e.what(), kDesc[cfg]);
    return 2;
  }
  fprintf(stderr, "[cfg%d %s] tile %dx%dx%d stages=%d smem=%zu B swizzle=%d raster=%s\n", cfg, kDesc[cfg], h->tile_m(), h->tile_n(),
          h->tile_k(), h->stages(), h->smem_bytes(), swizzle, ras.c_str());

  // Correctness: rows [0, r0) and the last r1 rows (covers the M tail) against the naive reference, alpha = sa*sb.
  std::string vstr = "SKIP";
  double rel_l2 = 0;
  if (verify) {
    int r0 = std::min(M, 512), r1 = std::min(M, 128);
    int start1 = M - r1;
    float* ref = nullptr;
    CUDA_OK(cudaMalloc(&ref, (size_t)std::max(r0, r1) * N * sizeof(float)));
    bench::ErrStats e0, e1;
    auto run_ref = [&](int row0, int rows) {
      float alpha = rowcol ? 1.f : sa * sb;
      if (is_int8)
        bench::ref_gemm_int8((int8_t*)A + (size_t)row0 * K, (int8_t*)B, ref, rows, N, K, alpha);
      else
        bench::ref_gemm_e4m3((__nv_fp8_e4m3*)A + (size_t)row0 * K, (__nv_fp8_e4m3*)B, ref, rows, N, K, alpha);
      if (rowcol) {
        size_t n = (size_t)rows * N;
        scale_rowcol_kernel<<<(unsigned)((n + 255) / 256), 256>>>(ref, rows, N, d_sa_vec + row0, d_sb_vec);
      }
      CUDA_OK(cudaDeviceSynchronize());
      return bench::compare_bf16_vs_f32(D + (size_t)row0 * N, ref, (size_t)rows * N);
    };
    e0 = run_ref(0, r0);
    e1 = run_ref(start1, r1);
    rel_l2 = std::max(e0.rel_l2, e1.rel_l2);
    bool pass = rel_l2 < 3e-3 && e0.ref_max_abs > 0;
    vstr = pass ? "PASS" : "FAIL";
    fprintf(stderr, "verify rows[0,%d) rel_l2=%.3e max_abs=%.3e ref_max=%.3e | rows[%d,%d) rel_l2=%.3e max_abs=%.3e\n", r0, e0.rel_l2,
            e0.max_abs, e0.ref_max_abs, start1, M, e1.rel_l2, e1.max_abs);
    CUDA_OK(cudaFree(ref));
  }

  bench::L2Flush* fl = flush ? new bench::L2Flush() : nullptr;
  int it = 0;
  auto t = bench::time_kernel([&] { hs[(it++) % nw]->run(0); }, warmup, iters, fl, 0, warmup_ms);
  delete fl;
  if (!dump.empty()) {
    FILE* f = fopen(dump.c_str(), "w");
    if (f) { for (double v : t.all_us) fprintf(f, "%.2f\n", v); fclose(f); }
  }
  double flop = 2.0 * M * N * K;
  printf("lib=cutlass\tdtype=%s\tcfg=%d\tM=%d\tN=%d\tK=%d\tmedian_us=%.2f\tmean_us=%.2f\tmin_us=%.2f\ttflops=%.1f\tclk_before=%d\tclk_after=%d\t"
         "verify=%s rel_l2=%.2e\tswizzle=%d\traster=%s\tflush=%d\tnw=%d\tdist=%s\twarmup_ms=%.0f\ttj_c=%.1f\tscale=%s\tdesc=%s\n",
         dtype.c_str(), cfg, M, N, K, t.median_us, t.mean_us, t.min_us, bench::tflops(flop, t.median_us), t.clock_mhz_before,
         t.clock_mhz_after, vstr.c_str(), rel_l2, swizzle, ras.c_str(), flush, nw, zeros ? "zeros" : dist.c_str(), warmup_ms,
         bench::gpu_temp_mc() / 1000.0, scale.c_str(), h->desc().c_str());
  hs.clear();
  CUDA_OK(cudaFree(A));
  CUDA_OK(cudaFree(Ball));
  CUDA_OK(cudaFree(D));
  CUDA_OK(cudaFree(d_scales));
  if (d_sa_vec) CUDA_OK(cudaFree(d_sa_vec));
  if (d_sb_vec) CUDA_OK(cudaFree(d_sb_vec));
  return vstr == "FAIL" ? 3 : 0;
}
