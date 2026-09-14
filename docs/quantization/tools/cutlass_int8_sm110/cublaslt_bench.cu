// cuBLASLt GEMM benchmark: the "vendor library" baseline for the CUTLASS INT8-vs-FP8 comparison on Thor (sm_110a).
//
//   D[M,N] = alpha * A[M,K] . W[N,K]^T        A, W row-major with K contiguous (W is an nn.Linear weight)
//
// In cuBLASLt column-major terms this is the "TN" case:
//   op(A_lt) = T,  A_lt = W viewed as a [K,N] col-major matrix, lda = K
//   op(B_lt) = N,  B_lt = A viewed as a [K,M] col-major matrix, ldb = K
//   D_lt is [N,M] col-major with ldc = N, which is exactly row-major D[M,N].
// 8-bit inputs (INT8, FP8) require exactly transa=T / transb=N in cuBLASLt, which this satisfies.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_110a -o cublaslt_bench cublaslt_bench.cu -lcublasLt -lcublas
// Usage:
//   ./cublaslt_bench --dtype=bf16|fp8|int8 [--m=4096 --n=4096 --k=4096 --iters=50 --warmup=10 --flush=1 --nw=1
//                    --verify=1 --autotune=0 --sa=0.0123 --sb=0.0456 --ws_mb=32]
//   --nw=N rotates through N distinct weight copies (one per timed iteration): --flush=0 --nw=8 models the production
//   layer loop on Thor (activations hot in L2, each layer's weight cold in DRAM); --flush=1 is all operands cold.
//
// Dtype paths:
//   bf16 : A/W BF16, D BF16, COMPUTE_32F, alpha = sa*sb (host float), beta = 0.
//   fp8  : A/W E4M3, D BF16, COMPUTE_32F, alpha = 1, per-tensor A/B scale pointers (device floats sa, sb).
//          Falls back to alpha = sa*sb without scale pointers if cuBLASLt rejects the scaled variant.
//   int8 : A/W INT8, D INT32, COMPUTE_32I, alpha = 1 (what torch._int_mm does; cuBLASLt has no INT8 scaling).
//          Also times "int8+rescale": the INT8 GEMM plus a fused int32 -> bf16 * (sa*sb) kernel, since that
//          epilogue is what INT8 needs to match the bf16-output FP8 path.
//
// stdout: exactly one tab-separated line per timed variant (see emit()). Everything else goes to stderr.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "bench_util.cuh"

#define LT_OK(call)                                                                                        \
  do {                                                                                                     \
    cublasStatus_t _s = (call);                                                                            \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                                                     \
      fprintf(stderr, "cuBLASLt error %s at %s:%d: status %d (%s)\n", #call, __FILE__, __LINE__, (int)_s, \
              cublasLtGetStatusString(_s));                                                                \
      exit(1);                                                                                             \
    }                                                                                                      \
  } while (0)

namespace {

// int32 accumulator -> bf16, multiplied by alpha. 4 elements per thread (int4 load, uint2 store); scalar tail.
__global__ void i32_to_bf16_scale_kernel(const int32_t* __restrict__ in, __nv_bfloat16* __restrict__ out, size_t n,
                                         float alpha) {
  size_t i4 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t base = i4 * 4;
  if (base + 4 <= n) {
    int4 v = reinterpret_cast<const int4*>(in)[i4];
    __nv_bfloat162 lo = __floats2bfloat162_rn((float)v.x * alpha, (float)v.y * alpha);
    __nv_bfloat162 hi = __floats2bfloat162_rn((float)v.z * alpha, (float)v.w * alpha);
    uint2 packed;
    packed.x = *reinterpret_cast<uint32_t*>(&lo);
    packed.y = *reinterpret_cast<uint32_t*>(&hi);
    reinterpret_cast<uint2*>(out)[i4] = packed;
  } else {
    for (size_t j = base; j < n; ++j) out[j] = __float2bfloat16((float)in[j] * alpha);
  }
}

void i32_to_bf16_scale(const int32_t* in, __nv_bfloat16* out, size_t n, float alpha, cudaStream_t s) {
  size_t n4 = (n + 3) / 4;
  unsigned blocks = (unsigned)((n4 + 255) / 256);
  i32_to_bf16_scale_kernel<<<blocks, 256, 0, s>>>(in, out, n, alpha);
  CUDA_OK(cudaGetLastError());
}

// Attribute widths differ (int32/uint32 for most, uint16 for INNER_SHAPE_ID / CLUSTER_SHAPE_ID), so query the
// required size first and read into a zero-initialized buffer of exactly that size.
int algo_cfg(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
  size_t need = 0, written = 0;
  LT_OK(cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, nullptr, 0, &need));
  uint64_t v = 0;
  if (need == 0 || need > sizeof(v)) return -1;
  LT_OK(cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &v, need, &written));
  return (int)v;  // little-endian: the low bytes hold the value regardless of width
}
int algo_id(const cublasLtMatmulAlgo_t& algo) { return algo_cfg(algo, CUBLASLT_ALGO_CONFIG_ID); }

// Human-readable algo config for stderr diagnostics (several heuristic candidates typically share one algo id and
// differ only in tile / stages / split-K / cluster shape).
std::string algo_desc(const cublasLtMatmulAlgo_t& algo) {
  char buf[128];
  snprintf(buf, sizeof(buf), "id %d tile %d stages %d splitk %d swz %d cluster %d", algo_id(algo),
           algo_cfg(algo, CUBLASLT_ALGO_CONFIG_TILE_ID), algo_cfg(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID),
           algo_cfg(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM), algo_cfg(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
           algo_cfg(algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID));
  return buf;
}

// Heuristic candidates whose state is SUCCESS. Empty on failure; *st carries the cuBLASLt status.
std::vector<cublasLtMatmulHeuristicResult_t> get_heuristics(cublasLtHandle_t h, cublasLtMatmulDesc_t desc,
                                                            cublasLtMatrixLayout_t la, cublasLtMatrixLayout_t lb,
                                                            cublasLtMatrixLayout_t lc, cublasLtMatrixLayout_t ld,
                                                            cublasLtMatmulPreference_t pref, int requested,
                                                            cublasStatus_t* st) {
  std::vector<cublasLtMatmulHeuristicResult_t> res(requested);
  int returned = 0;
  *st = cublasLtMatmulAlgoGetHeuristic(h, desc, la, lb, lc, ld, pref, requested, res.data(), &returned);
  if (*st != CUBLAS_STATUS_SUCCESS) returned = 0;
  res.resize(returned);
  std::vector<cublasLtMatmulHeuristicResult_t> ok;
  for (auto& r : res)
    if (r.state == CUBLAS_STATUS_SUCCESS) ok.push_back(r);
  return ok;
}

void usage() {
  fprintf(stderr,
          "usage: cublaslt_bench --dtype=bf16|fp8|int8 [--m=4096 --n=4096 --k=4096 --iters=50 --warmup=10 --flush=1 "
          "--verify=1 --autotune=0 --sa=0.0123 --sb=0.0456 --ws_mb=32]\n");
}

}  // namespace

int main(int argc, char** argv) {
  bench::Args args(argc, argv);
  const std::string dtype = args.get("dtype", "");
  if (!(dtype == "bf16" || dtype == "fp8" || dtype == "int8")) {
    usage();
    return 2;
  }
  const int M = args.geti("m", 4096), N = args.geti("n", 4096), K = args.geti("k", 4096);
  const int iters = args.geti("iters", 50), warmup = args.geti("warmup", 10);
  const bool flush = args.geti("flush", 1) != 0, verify = args.geti("verify", 1) != 0;
  const bool autotune = args.geti("autotune", 0) != 0;
  const int nw = std::max(1, args.geti("nw", 1));
  const double warmup_ms = std::stod(args.get("warmup_ms", "300"));  // min warm-up wall time before timing (1500+ = sustained/power-limited regime)
  const float sa = std::stof(args.get("sa", "0.0123")), sb = std::stof(args.get("sb", "0.0456"));
  const size_t ws_bytes = (size_t)args.geti("ws_mb", 32) << 20;
  const float alpha_sab = sa * sb;
  if (M <= 0 || N <= 0 || K <= 0 || iters <= 0) {
    usage();
    return 2;
  }

  const bool is_bf16 = dtype == "bf16", is_fp8 = dtype == "fp8", is_int8 = dtype == "int8";
  const cudaDataType_t type_in = is_bf16 ? CUDA_R_16BF : is_fp8 ? CUDA_R_8F_E4M3 : CUDA_R_8I;
  const cudaDataType_t type_out = is_int8 ? CUDA_R_32I : CUDA_R_16BF;
  const cublasComputeType_t compute = is_int8 ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F;
  const cudaDataType_t type_scale = is_int8 ? CUDA_R_32I : CUDA_R_32F;
  const size_t in_bytes = is_bf16 ? 2 : 1, out_bytes = is_int8 ? 4 : 2;

  cudaDeviceProp prop;
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  fprintf(stderr,
          "[cublaslt_bench] %s (sm_%d%d, %d SMs) cuBLASLt %zu | dtype=%s M=%d N=%d K=%d iters=%d warmup=%d flush=%d "
          "verify=%d autotune=%d sa=%g sb=%g ws=%zu MB\n",
          prop.name, prop.major, prop.minor, prop.multiProcessorCount, cublasLtGetVersion(), dtype.c_str(), M, N, K,
          iters, warmup, (int)flush, (int)verify, (int)autotune, sa, sb, ws_bytes >> 20);

  // ------------------------------------------------------------------ operands
  const size_t nA = (size_t)M * K, nW = (size_t)N * K, nD = (size_t)M * N;
  void *dA = nullptr, *dW = nullptr, *dD = nullptr, *ws = nullptr;
  __nv_bfloat16* dDbf = nullptr;  // int8+rescale output
  float *d_sa = nullptr, *d_sb = nullptr;
  CUDA_OK(cudaMalloc(&dA, nA * in_bytes));
  CUDA_OK(cudaMalloc(&dW, nW * in_bytes * nw));  // nw weight copies back to back; copy 0 is used for verification
  CUDA_OK(cudaMalloc(&dD, nD * out_bytes));
  if (is_int8) CUDA_OK(cudaMalloc(&dDbf, nD * sizeof(__nv_bfloat16)));
  if (ws_bytes) CUDA_OK(cudaMalloc(&ws, ws_bytes));
  CUDA_OK(cudaMalloc(&d_sa, sizeof(float)));
  CUDA_OK(cudaMalloc(&d_sb, sizeof(float)));
  CUDA_OK(cudaMemcpy(d_sa, &sa, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_sb, &sb, sizeof(float), cudaMemcpyHostToDevice));

  for (int w = 0; w < nw; ++w) {
    char* wp = (char*)dW + (size_t)w * nW * in_bytes;
    const uint32_t seed = 0x5678u + 977u * w;
    if (is_bf16) bench::fill_bf16((__nv_bfloat16*)wp, nW, seed);
    else if (is_fp8) bench::fill_e4m3((__nv_fp8_e4m3*)wp, nW, seed);
    else bench::fill_int8((int8_t*)wp, nW, seed);
  }
  if (is_bf16) bench::fill_bf16((__nv_bfloat16*)dA, nA, 0x1234u);
  else if (is_fp8) bench::fill_e4m3((__nv_fp8_e4m3*)dA, nA, 0x1234u);
  else bench::fill_int8((int8_t*)dA, nA, 0x1234u);
  CUDA_OK(cudaMemset(dD, 0, nD * out_bytes));
  CUDA_OK(cudaDeviceSynchronize());

  // ------------------------------------------------------------------ cuBLASLt objects
  cublasLtHandle_t handle;
  LT_OK(cublasLtCreate(&handle));
  cublasLtMatmulDesc_t desc;
  LT_OK(cublasLtMatmulDescCreate(&desc, compute, type_scale));
  const cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
  LT_OK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
  LT_OK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

  // Layouts are column-major (rows, cols, ld). See header comment for the row-major <-> col-major mapping.
  cublasLtMatrixLayout_t la, lb, lc, ld;
  LT_OK(cublasLtMatrixLayoutCreate(&la, type_in, K, N, K));   // A_lt = W : [K,N] cm == W[N,K] rm
  LT_OK(cublasLtMatrixLayoutCreate(&lb, type_in, K, M, K));   // B_lt = A : [K,M] cm == A[M,K] rm
  LT_OK(cublasLtMatrixLayoutCreate(&lc, type_out, N, M, N));  // C     : [N,M] cm == D[M,N] rm
  LT_OK(cublasLtMatrixLayoutCreate(&ld, type_out, N, M, N));  // D     : [N,M] cm == D[M,N] rm

  cublasLtMatmulPreference_t pref;
  LT_OK(cublasLtMatmulPreferenceCreate(&pref));
  LT_OK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
                                             sizeof(ws_bytes)));

  // alpha/beta live in host memory (default CUBLASLT_POINTER_MODE_HOST).
  float alpha_f = is_fp8 ? 1.f : alpha_sab, beta_f = 0.f;
  int32_t alpha_i = 1, beta_i = 0;
  const void* alpha_p = is_int8 ? (const void*)&alpha_i : (const void*)&alpha_f;
  const void* beta_p = is_int8 ? (const void*)&beta_i : (const void*)&beta_f;

  // FP8 per-tensor scaling: cuBLASLt's A operand is our W (scale sb), its B operand is our A (scale sa).
  bool fp8_scale_ptrs = is_fp8;
  auto set_scale_ptrs = [&](bool on) {
    const float* pa = on ? d_sb : nullptr;
    const float* pb = on ? d_sa : nullptr;
    LT_OK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa, sizeof(pa)));
    LT_OK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb, sizeof(pb)));
  };
  if (is_fp8) set_scale_ptrs(true);

  auto run_w = [&](const cublasLtMatmulAlgo_t* algo, int w) -> cublasStatus_t {
    const void* wp = (const char*)dW + (size_t)w * nW * in_bytes;
    return cublasLtMatmul(handle, desc, alpha_p, wp, la, dA, lb, beta_p, dD, lc, dD, ld, algo, ws, ws_bytes,
                          /*stream=*/0);
  };
  auto run = [&](const cublasLtMatmulAlgo_t* algo) -> cublasStatus_t { return run_w(algo, 0); };

  // ------------------------------------------------------------------ algorithm selection (with FP8 fallback)
  const int requested = autotune ? 16 : 1;
  std::vector<cublasLtMatmulHeuristicResult_t> cands;
  for (;;) {
    cublasStatus_t hst;
    cands = get_heuristics(handle, desc, la, lb, lc, ld, pref, requested, &hst);
    cublasStatus_t trial = cands.empty() ? CUBLAS_STATUS_NOT_SUPPORTED : run(&cands[0].algo);
    if (trial == CUBLAS_STATUS_SUCCESS) {
      cudaError_t ce = cudaDeviceSynchronize();
      if (ce != cudaSuccess) {
        fprintf(stderr, "CUDA error after trial cublasLtMatmul: %s\n", cudaGetErrorString(ce));
        return 1;
      }
      break;
    }
    if (fp8_scale_ptrs) {
      fprintf(stderr,
              "NOTE: cuBLASLt rejected FP8 with per-tensor A/B scale pointers on this platform (heuristic status %d "
              "(%s), %zu candidates, matmul status %d (%s)). Falling back to alpha=sa*sb without scale pointers.\n",
              (int)hst, cublasLtGetStatusString(hst), cands.size(), (int)trial, cublasLtGetStatusString(trial));
      fp8_scale_ptrs = false;
      set_scale_ptrs(false);
      alpha_f = alpha_sab;
      continue;
    }
    fprintf(stderr, "cuBLASLt: no usable algorithm for dtype=%s M=%d N=%d K=%d (heuristic status %d (%s), %zu "
            "candidates, matmul status %d (%s))\n",
            dtype.c_str(), M, N, K, (int)hst, cublasLtGetStatusString(hst), cands.size(), (int)trial,
            cublasLtGetStatusString(trial));
    return 1;
  }
  if (is_fp8) fprintf(stderr, "fp8 scaling: %s\n", fp8_scale_ptrs ? "A/B scale pointers (alpha=1)" : "alpha=sa*sb");

  std::unique_ptr<bench::L2Flush> flusher(flush ? new bench::L2Flush() : nullptr);
  const bench::L2Flush* flush_p = flusher.get();

  int chosen = 0;
  std::string algo_str = "heuristic0";
  fprintf(stderr, "heuristic returned %zu candidate(s); heuristic0: %s ws %zu B\n", cands.size(),
          algo_desc(cands[0].algo).c_str(), cands[0].workspaceSize);
  if (autotune) {
    double best = 1e300;
    for (size_t i = 0; i < cands.size(); ++i) {
      cublasStatus_t st = run(&cands[i].algo);
      if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "  autotune cand %zu: %s -> status %d (%s), skipped\n", i, algo_desc(cands[i].algo).c_str(),
                (int)st, cublasLtGetStatusString(st));
        continue;
      }
      // rank candidates in the same regime as the measurement: same flush setting, same weight rotation, 20 iterations, short warm-up
      int itc = 0;
      auto r = bench::time_kernel([&] { LT_OK(run_w(&cands[i].algo, (itc++) % nw)); }, 2, 20, flush_p, 0, 50.0);
      fprintf(stderr, "  autotune cand %zu: %s ws %zu B waves %.2f -> median %.2f us\n", i,
              algo_desc(cands[i].algo).c_str(), cands[i].workspaceSize, cands[i].wavesCount, r.median_us);
      if (r.median_us < best) {
        best = r.median_us;
        chosen = (int)i;
      }
    }
    algo_str = std::to_string(algo_id(cands[chosen].algo));
    fprintf(stderr, "autotune picked cand %d (%s)\n", chosen, algo_desc(cands[chosen].algo).c_str());
  }
  const cublasLtMatmulAlgo_t* algo = &cands[chosen].algo;

  // ------------------------------------------------------------------ timing
  int it = 0;
  auto t_gemm = bench::time_kernel([&] { LT_OK(run_w(algo, (it++) % nw)); }, warmup, iters, flush_p, 0, warmup_ms);
  bench::TimingResult t_rescale;
  if (is_int8) {
    it = 0;
    t_rescale = bench::time_kernel(
        [&] {
          LT_OK(run_w(algo, (it++) % nw));
          i32_to_bf16_scale((const int32_t*)dD, dDbf, nD, alpha_sab, 0);
        },
        warmup, iters, flush_p, 0, warmup_ms);
  }
  CUDA_OK(cudaDeviceSynchronize());
  // leave D / Dbf computed from weight copy 0 for the verification below
  LT_OK(run(algo));
  if (is_int8) i32_to_bf16_scale((const int32_t*)dD, dDbf, nD, alpha_sab, 0);
  CUDA_OK(cudaDeviceSynchronize());

  // ------------------------------------------------------------------ verification
  std::string v_gemm = "SKIP", v_res = "SKIP";
  double rl_gemm = 0, rl_res = 0;
  if (verify) {
    // For large M only check rows [0,512) and the last 128 rows; gather them contiguously so one compare gives a
    // single combined rel-L2.
    std::vector<std::pair<int, int>> ranges;  // (row0, rows)
    if (M > 1024) ranges = {{0, 512}, {M - 128, 128}};
    else ranges = {{0, M}};
    int rows_chk = 0;
    for (auto& r : ranges) rows_chk += r.second;
    const size_t n_chk = (size_t)rows_chk * N;
    fprintf(stderr, "verify: %d rows (%s)\n", rows_chk, M > 1024 ? "[0,512) + last 128" : "all");

    float* dRef = nullptr;
    void* dChk = nullptr;
    __nv_bfloat16* dChkBf = nullptr;
    CUDA_OK(cudaMalloc(&dRef, n_chk * sizeof(float)));
    CUDA_OK(cudaMalloc(&dChk, n_chk * out_bytes));
    if (is_int8) CUDA_OK(cudaMalloc(&dChkBf, n_chk * sizeof(__nv_bfloat16)));

    auto gather = [&](const void* src, void* dst, size_t eb) {
      int off = 0;
      for (auto& r : ranges) {
        CUDA_OK(cudaMemcpy((char*)dst + (size_t)off * N * eb, (const char*)src + (size_t)r.first * N * eb,
                           (size_t)r.second * N * eb, cudaMemcpyDeviceToDevice));
        off += r.second;
      }
    };
    auto reference = [&](float alpha) {
      int off = 0;
      for (auto& r : ranges) {
        float* dst = dRef + (size_t)off * N;
        const size_t a_off = (size_t)r.first * K;
        if (is_bf16)
          bench::ref_gemm_bf16((const __nv_bfloat16*)dA + a_off, (const __nv_bfloat16*)dW, dst, r.second, N, K, alpha);
        else if (is_fp8)
          bench::ref_gemm_e4m3((const __nv_fp8_e4m3*)dA + a_off, (const __nv_fp8_e4m3*)dW, dst, r.second, N, K, alpha);
        else
          bench::ref_gemm_int8((const int8_t*)dA + a_off, (const int8_t*)dW, dst, r.second, N, K, alpha);
        off += r.second;
      }
      CUDA_OK(cudaDeviceSynchronize());
    };

    gather(dD, dChk, out_bytes);
    if (is_int8) {
      reference(1.f);
      auto e = bench::compare_i32_vs_f32((const int32_t*)dChk, dRef, n_chk);
      // fp32 reference is exact while |sum| < 2^24; beyond that allow float(int32) rounding.
      const bool exact = (double)K * 127.0 * 127.0 < 16777216.0;
      const double tol = exact ? 0.0 : 1e-6;
      rl_gemm = e.rel_l2;
      v_gemm = (e.rel_l2 <= tol) ? "PASS" : "FAIL";
      fprintf(stderr, "verify int8: rel_l2 %.3e max_abs %.3g ref_max_abs %.3g (tol %s) -> %s\n", e.rel_l2, e.max_abs,
              e.ref_max_abs, exact ? "exact" : "1e-6", v_gemm.c_str());

      gather(dDbf, dChkBf, sizeof(__nv_bfloat16));
      reference(alpha_sab);
      auto e2 = bench::compare_bf16_vs_f32(dChkBf, dRef, n_chk);
      rl_res = e2.rel_l2;
      v_res = (e2.rel_l2 < 3e-3) ? "PASS" : "FAIL";
      fprintf(stderr, "verify int8+rescale: rel_l2 %.3e max_abs %.3g ref_max_abs %.3g -> %s\n", e2.rel_l2, e2.max_abs,
              e2.ref_max_abs, v_res.c_str());
    } else {
      reference(alpha_sab);
      auto e = bench::compare_bf16_vs_f32((const __nv_bfloat16*)dChk, dRef, n_chk);
      rl_gemm = e.rel_l2;
      v_gemm = (e.rel_l2 < 3e-3) ? "PASS" : "FAIL";
      fprintf(stderr, "verify %s: rel_l2 %.3e max_abs %.3g ref_max_abs %.3g -> %s\n", dtype.c_str(), e.rel_l2,
              e.max_abs, e.ref_max_abs, v_gemm.c_str());
    }
    CUDA_OK(cudaFree(dRef));
    CUDA_OK(cudaFree(dChk));
    if (dChkBf) CUDA_OK(cudaFree(dChkBf));
  }

  // ------------------------------------------------------------------ output (stdout, machine-parsable)
  const double flop = 2.0 * (double)M * (double)N * (double)K;
  auto emit = [&](const char* name, const bench::TimingResult& t, const std::string& v, double rl) {
    printf("lib=cublaslt\tdtype=%s\tM=%d\tN=%d\tK=%d\tmedian_us=%.2f\tmean_us=%.2f\tmin_us=%.2f\ttflops=%.1f\t"
           "clk_before=%d\tclk_after=%d\tverify=%s rel_l2=%.2e\tflush=%d\tnw=%d\twarmup_ms=%.0f\talgo=%s\n",
           name, M, N, K, t.median_us, t.mean_us, t.min_us, bench::tflops(flop, t.median_us), t.clock_mhz_before,
           t.clock_mhz_after, v.c_str(), rl, (int)flush, nw, warmup_ms, algo_str.c_str());
  };
  emit(dtype.c_str(), t_gemm, v_gemm, rl_gemm);
  if (is_int8) emit("int8+rescale", t_rescale, v_res, rl_res);
  fflush(stdout);

  // ------------------------------------------------------------------ cleanup
  LT_OK(cublasLtMatmulPreferenceDestroy(pref));
  LT_OK(cublasLtMatrixLayoutDestroy(la));
  LT_OK(cublasLtMatrixLayoutDestroy(lb));
  LT_OK(cublasLtMatrixLayoutDestroy(lc));
  LT_OK(cublasLtMatrixLayoutDestroy(ld));
  LT_OK(cublasLtMatmulDescDestroy(desc));
  LT_OK(cublasLtDestroy(handle));
  flusher.reset();
  CUDA_OK(cudaFree(dA));
  CUDA_OK(cudaFree(dW));
  CUDA_OK(cudaFree(dD));
  if (dDbf) CUDA_OK(cudaFree(dDbf));
  if (ws) CUDA_OK(cudaFree(ws));
  CUDA_OK(cudaFree(d_sa));
  CUDA_OK(cudaFree(d_sb));

  const bool failed = v_gemm == "FAIL" || v_res == "FAIL";
  return failed ? 1 : 0;
}
