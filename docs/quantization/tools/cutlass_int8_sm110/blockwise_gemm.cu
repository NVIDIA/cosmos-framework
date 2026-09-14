// Benchmark / correctness driver for the block-scaled (g128) FP8 and INT8 GEMMs on Thor (sm_110a).
// Usage: ./blockwise_gemm --dtype=fp8|int8 --cfg=1 --m=1517 --n=4096 --k=4096 [--iters=50 --warmup=10 --warmup_ms=1500 --flush=0 --nw=8
//                         --verify=1 --dist=uniform|normal --zeros=0 --swizzle=auto|<int> --raster=H|M|N --dump=file]   |  --list
// Scales: sfa[m, kb] and sfb[n or n/128, kb] fp32 in [0.5e-2, 2e-2], CUTLASS MN-major layout ([K/GK][ceil(M/GM)], [K/GK][ceil(N/GN)]).
// Output line: lib=cutlass_bw dtype cfg gran M N K median_us mean_us min_us tflops clk verify rel_l2 swizzle raster flush nw dist warmup_ms tj_c desc
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "bench_util.cuh"
#include "blockwise_gemm.cuh"
#include "bw_configs.h"

#define DECL_F8(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) THOR_BW_DECL(fp8, I)
THOR_BW_CFG_LIST(DECL_F8)
#ifdef THOR_BW_HAVE_INT8
#define DECL_I8(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) THOR_BW_DECL(int8, I)
THOR_BW_CFG_LIST(DECL_I8)
#endif

using Factory = thor::BwHandle* (*)();
static Factory kFp8[] = {
#define F_F8(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) thor_bw_make_fp8_##I,
    THOR_BW_CFG_LIST(F_F8)
};
#ifdef THOR_BW_HAVE_INT8
static Factory kInt8[] = {
#define F_I8(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) thor_bw_make_int8_##I,
    THOR_BW_CFG_LIST(F_I8)
};
#endif
static const char* kDesc[] = {
#define D_(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) D,
    THOR_BW_CFG_LIST(D_)
};

int main(int argc, char** argv) {
  bench::Args args(argc, argv);
  if (args.has("list")) {
    for (int i = 0; i < THOR_NUM_BW_CFGS; ++i) printf("cfg%d: %s\n", i, kDesc[i]);
    return 0;
  }
  std::string dtype = args.get("dtype", "fp8");
  int cfg = args.geti("cfg", 1), M = args.geti("m", 1517), N = args.geti("n", 4096), K = args.geti("k", 4096);
  int iters = args.geti("iters", 50), warmup = args.geti("warmup", 10), flush = args.geti("flush", 0);
  int verify = args.geti("verify", 1), nw = std::max(1, args.geti("nw", 8)), zeros = args.geti("zeros", 0);
  std::string swz = args.get("swizzle", "auto"), ras = args.get("raster", "H"), dist = args.get("dist", "normal"), dump = args.get("dump", "");
  double warmup_ms = std::stod(args.get("warmup_ms", "1500"));
  int raster = ras == "M" ? 1 : ras == "N" ? 2 : 0;
  if (cfg < 0 || cfg >= THOR_NUM_BW_CFGS) { fprintf(stderr, "bad cfg\n"); return 1; }
  bool is_int8 = dtype == "int8";
  if (!is_int8 && dtype != "fp8") { fprintf(stderr, "dtype must be int8|fp8\n"); return 1; }
#ifndef THOR_BW_HAVE_INT8
  if (is_int8) { fprintf(stderr, "int8 blockwise not compiled in\n"); return 1; }
#endif

  std::vector<std::unique_ptr<thor::BwHandle>> hs;
  for (int w = 0; w < nw; ++w) {
#ifdef THOR_BW_HAVE_INT8
    hs.emplace_back((is_int8 ? kInt8 : kFp8)[cfg]());
#else
    hs.emplace_back(kFp8[cfg]());
#endif
  }
  thor::BwHandle* h = hs[0].get();
  const int GM = h->gran_m(), GN = h->gran_n(), GK = h->gran_k();
  if (K % GK != 0) { fprintf(stderr, "K must be a multiple of %d\n", GK); return 1; }
  const int KB = K / GK, MB = (M + GM - 1) / GM, NB = (N + GN - 1) / GN;
  if (h->sf_bytes() != 4) { fprintf(stderr, "scale element is %d bytes (not fp32): this instantiation is unusable\n", h->sf_bytes()); return 1; }

  size_t nA = (size_t)M * K, nB = (size_t)N * K, nD = (size_t)M * N, nSFA = (size_t)KB * MB, nSFB = (size_t)KB * NB;
  void *A = nullptr, *Ball = nullptr;
  float *sfa = nullptr, *sfball = nullptr;
  __nv_bfloat16* D = nullptr;
  CUDA_OK(cudaMalloc(&A, nA));
  CUDA_OK(cudaMalloc(&Ball, nB * nw));
  CUDA_OK(cudaMalloc(&sfa, nSFA * sizeof(float)));
  CUDA_OK(cudaMalloc(&sfball, nSFB * sizeof(float) * nw));
  CUDA_OK(cudaMalloc(&D, nD * sizeof(__nv_bfloat16)));
  if (zeros) {
    CUDA_OK(cudaMemset(A, 0, nA)); CUDA_OK(cudaMemset(Ball, 0, nB * nw)); verify = 0;
  } else if (is_int8) {
    if (dist == "normal") { bench::fill_int8_normal((int8_t*)A, nA, 0x1234u); for (int w = 0; w < nw; ++w) bench::fill_int8_normal((int8_t*)Ball + (size_t)w * nB, nB, 0x5678u + 977u * w); }
    else { bench::fill_int8((int8_t*)A, nA, 0x1234u); for (int w = 0; w < nw; ++w) bench::fill_int8((int8_t*)Ball + (size_t)w * nB, nB, 0x5678u + 977u * w); }
  } else {
    if (dist == "normal") { bench::fill_e4m3_normal((__nv_fp8_e4m3*)A, nA, 0x1234u); for (int w = 0; w < nw; ++w) bench::fill_e4m3_normal((__nv_fp8_e4m3*)Ball + (size_t)w * nB, nB, 0x5678u + 977u * w); }
    else { bench::fill_e4m3((__nv_fp8_e4m3*)A, nA, 0x1234u); for (int w = 0; w < nw; ++w) bench::fill_e4m3((__nv_fp8_e4m3*)Ball + (size_t)w * nB, nB, 0x5678u + 977u * w); }
  }
  bench::fill_scales(sfa, nSFA, 0xabcdu);   // for the separable kernel these are sfa'[m,g] = sfa[m,g]*c[g] (folded at quantization time)
  for (int w = 0; w < nw; ++w) bench::fill_scales(sfball + (size_t)w * nSFB, nSFB, 0xdcbau + 31u * w);
  float* colscale = nullptr;  // separable variant: per-output-channel factor s_w[n], applied in the epilogue
  if (h->colscale_epi()) {
    CUDA_OK(cudaMalloc(&colscale, (size_t)N * nw * sizeof(float)));
    for (int w = 0; w < nw; ++w) bench::fill_scales(colscale + (size_t)w * N, N, 0x7777u + 13u * w, 0.5f, 2.0f);
  }
  CUDA_OK(cudaMemset(D, 0, nD * sizeof(__nv_bfloat16)));
  CUDA_OK(cudaDeviceSynchronize());

  int swizzle = 1;
  if (swz == "auto") {
    if ((double)N * K > 16.0 * 1024 * 1024) { double budget = 16.0 * 1024 * 1024 / ((double)h->tile_m() * K); while (swizzle * 2 <= budget && swizzle < 16) swizzle *= 2; }
  } else swizzle = std::stoi(swz);
  try {
    for (int w = 0; w < nw; ++w)
      hs[w]->init(A, (char*)Ball + (size_t)w * nB, sfa, sfball + (size_t)w * nSFB, D, M, N, K, swizzle, raster, 0,
                  colscale ? colscale + (size_t)w * N : nullptr);
    h->run(0);
    CUDA_OK(cudaDeviceSynchronize());
  } catch (std::exception const& e) {
    printf("lib=cutlass_bw\tdtype=%s\tcfg=%d\tM=%d\tN=%d\tK=%d\tERROR=%s\tdesc=%s\n", dtype.c_str(), cfg, M, N, K, e.what(), kDesc[cfg]);
    return 2;
  }
  fprintf(stderr, "[bw cfg%d %s] tile %dx%dx%d gran %dx%dx%d stages=%d smem=%zu B swizzle=%d raster=%s\n", cfg, kDesc[cfg], h->tile_m(),
          h->tile_n(), h->tile_k(), GM, GN, GK, h->stages(), h->smem_bytes(), swizzle, ras.c_str());

  std::string vstr = "SKIP";
  double rel_l2 = 0;
  if (verify) {
    int r0 = std::min(M, 512), r1 = std::min(M, 128), start1 = M - r1;
    float* ref = nullptr;
    CUDA_OK(cudaMalloc(&ref, (size_t)std::max(r0, r1) * N * sizeof(float)));
    auto run_ref = [&](int row0, int rows) {
      // rows [row0, row0+rows): pass A/sfa offsets; the reference indexes sfa by m/GM, so shift by row0 (GM == 1 here; general: row0 % GM == 0 required)
      if (is_int8)
        bench::ref_gemm_blockscaled<int8_t, int32_t>((int8_t*)A + (size_t)row0 * K, (int8_t*)Ball, sfa + row0 / GM, sfball, ref, rows, N, K, GM, GN, GK, MB, NB, colscale);
      else
        bench::ref_gemm_blockscaled<__nv_fp8_e4m3, float>((__nv_fp8_e4m3*)A + (size_t)row0 * K, (__nv_fp8_e4m3*)Ball, sfa + row0 / GM, sfball, ref, rows, N, K, GM, GN, GK, MB, NB, colscale);
      CUDA_OK(cudaDeviceSynchronize());
      return bench::compare_bf16_vs_f32(D + (size_t)row0 * N, ref, (size_t)rows * N);
    };
    auto e0 = run_ref(0, r0);
    auto e1 = run_ref(start1 - start1 % GM, r1 + start1 % GM);
    rel_l2 = std::max(e0.rel_l2, e1.rel_l2);
    bool pass = rel_l2 < 3e-3 && e0.ref_max_abs > 0;
    vstr = pass ? "PASS" : "FAIL";
    fprintf(stderr, "verify rows[0,%d) rel_l2=%.3e max_abs=%.3e ref_max=%.3e | tail rel_l2=%.3e max_abs=%.3e\n", r0, e0.rel_l2, e0.max_abs, e0.ref_max_abs, e1.rel_l2, e1.max_abs);
    CUDA_OK(cudaFree(ref));
  }

  bench::L2Flush* fl = flush ? new bench::L2Flush() : nullptr;
  int it = 0;
  auto t = bench::time_kernel([&] { hs[(it++) % nw]->run(0); }, warmup, iters, fl, 0, warmup_ms);
  delete fl;
  if (!dump.empty()) { FILE* f = fopen(dump.c_str(), "w"); if (f) { for (double v : t.all_us) fprintf(f, "%.2f\n", v); fclose(f); } }
  double flop = 2.0 * M * N * K;
  printf("lib=cutlass_bw\tdtype=%s\tcfg=%d\tgran=%dx%dx%d%s\tM=%d\tN=%d\tK=%d\tmedian_us=%.2f\tmean_us=%.2f\tmin_us=%.2f\ttflops=%.1f\tclk_before=%d\tclk_after=%d\t"
         "verify=%s rel_l2=%.2e\tswizzle=%d\traster=%s\tflush=%d\tnw=%d\tdist=%s\twarmup_ms=%.0f\ttj_c=%.1f\tdesc=%s\n",
         dtype.c_str(), cfg, GM, GN, GK, h->colscale_epi() ? "+colscale" : "", M, N, K, t.median_us, t.mean_us, t.min_us, bench::tflops(flop, t.median_us), t.clock_mhz_before,
         t.clock_mhz_after, vstr.c_str(), rel_l2, swizzle, ras.c_str(), flush, nw, zeros ? "zeros" : dist.c_str(), warmup_ms,
         bench::gpu_temp_mc() / 1000.0, kDesc[cfg]);
  hs.clear();
  if (colscale) CUDA_OK(cudaFree(colscale));
  CUDA_OK(cudaFree(A)); CUDA_OK(cudaFree(Ball)); CUDA_OK(cudaFree(sfa)); CUDA_OK(cudaFree(sfball)); CUDA_OK(cudaFree(D));
  return vstr == "FAIL" ? 3 : 0;
}
